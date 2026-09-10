import tempfile
import tkinter
import unittest
from pathlib import Path

from insta360_frame_extractor_gui import (
    DEFAULT_MASK_BATCH_SIZE,
    MASK_BATCH_SIZE_LIMIT,
    Direction,
    DirectionSet,
    build_direction_filter_graph,
    build_extracted_image_glob,
    build_footprint_segments,
    build_mask_output_path,
    build_merged_ffmpeg_command,
    build_overlapping_tile_regions,
    build_output_image_path,
    build_output_pattern,
    build_shared_decode_head,
    build_tile_starts,
    build_binary_usage_mask,
    collect_segformer_excluded_label_groups,
    collect_segformer_excluded_label_ids,
    compute_expected_frame_count,
    compute_output_direction_index,
    compute_vertical_fov,
    compute_preview_box,
    direction_to_equirect_point,
    direction_index_width,
    fit_size_within_bounds,
    generate_ring_directions,
    is_ffmpeg_progress_line,
    load_image_bgr,
    mask_contains_excluded_region,
    parse_angle,
    parse_ffmpeg_time_seconds,
    parse_probability_threshold,
    preload_native_backends,
    resolve_mask_detail_preset,
    select_ffmpeg_error_detail,
)


class Insta360FrameExtractorGuiTests(unittest.TestCase):
    def test_direction_index_width_has_minimum_two_digits(self) -> None:
        self.assertEqual(direction_index_width(1), 2)
        self.assertEqual(direction_index_width(8), 2)
        self.assertEqual(direction_index_width(120), 3)

    def test_output_pattern_matches_requested_naming(self) -> None:
        pattern = build_output_pattern(Path("C:/output"), "clip", 7, 8)
        self.assertEqual(str(pattern).replace("\\", "/"), "C:/output/clip_%04d_07.jpg")

    def test_compute_output_direction_index_reverses_on_odd_frames(self) -> None:
        self.assertEqual(compute_output_direction_index(0, 10, 0, True), 0)
        self.assertEqual(compute_output_direction_index(0, 10, 1, True), 9)
        self.assertEqual(compute_output_direction_index(9, 10, 1, True), 0)

    def test_build_output_image_path_uses_reversed_direction_index_on_odd_frames(self) -> None:
        image_path = build_output_image_path(
            Path("C:/output"),
            "clip",
            1,
            0,
            10,
            reverse_on_odd_frames=True,
        )
        self.assertEqual(str(image_path).replace("\\", "/"), "C:/output/clip_0001_09.jpg")

    def test_build_mask_output_path_matches_requested_naming(self) -> None:
        mask_path = build_mask_output_path(Path("C:/output/clip_0000_07.jpg"))
        self.assertEqual(str(mask_path).replace("\\", "/"), "C:/output/clip_0000_07.jpg.mask.png")

    def test_collect_segformer_excluded_label_ids_finds_sky_person_car_tree(self) -> None:
        label_ids = collect_segformer_excluded_label_ids(
            {
                0: "background",
                1: "sky",
                2: "person",
                3: "tree",
                4: "car",
                5: "building;edifice",
            }
        )
        self.assertEqual(label_ids, {1, 2, 3, 4})

    def test_collect_segformer_excluded_label_groups_splits_categories(self) -> None:
        label_groups = collect_segformer_excluded_label_groups(
            {
                0: "background",
                1: "sky",
                2: "person",
                3: "car",
                4: "tree",
            }
        )
        self.assertEqual(label_groups["sky"], {1})
        self.assertEqual(label_groups["person"], {2})
        self.assertEqual(label_groups["car"], {3})
        self.assertEqual(label_groups["tree"], {4})

    def test_build_binary_usage_mask_uses_confidence_threshold(self) -> None:
        binary_mask = build_binary_usage_mask(
            [[0.8, 0.2], [0.9, 0.95]],
            0.7,
        )
        self.assertEqual(binary_mask[0, 0], 0)
        self.assertEqual(binary_mask[0, 1], 255)
        self.assertEqual(binary_mask[1, 0], 0)
        self.assertEqual(binary_mask[1, 1], 0)

    def test_mask_contains_excluded_region_detects_black_pixels(self) -> None:
        self.assertTrue(mask_contains_excluded_region([[255, 0], [255, 255]]))
        self.assertFalse(mask_contains_excluded_region([[255, 255], [255, 255]]))

    def test_resolve_mask_detail_preset_falls_back_to_default(self) -> None:
        self.assertEqual(resolve_mask_detail_preset("存在しない設定")["prioritize_detail"], False)

    def test_select_ffmpeg_error_detail_leads_with_the_root_cause(self) -> None:
        # ffmpeg は原因 -> 波及の順に出すので、最終行ではなく先頭側の該当行が真因。
        # 以下は実際に ffmpeg 8.0.1 で再現させた 3 パターンの末尾ログ。
        v360_out_of_range = [
            "[Parsed_v360_1 @ 000] Value 225.000000 for parameter 'yaw' out of range [-180 - 180]",
            "[Parsed_v360_1 @ 000] Error setting option yaw to value 225.",
            "[vf#0:0 @ 000] Error initializing filter 'v360'",
            "Error : Result too large",
        ]
        detail = select_ffmpeg_error_detail(v360_out_of_range, "[1/6]", 1)
        self.assertIn("out of range", detail)
        self.assertTrue(detail.startswith("[Parsed_v360_1"))

        bad_hwaccel_device = [
            "[hevc @ 000] CUDA_ERROR_INVALID_DEVICE: invalid device ordinal",
            "[hevc @ 000] Failed setup for format cuda: hwaccel initialisation returned error.",
            "[vist#0:0 @ 000] Decoding error: Generic error in an external library",
            "Error binding filtergraph inputs/outputs: Generic error in an external library",
        ]
        detail = select_ffmpeg_error_detail(bad_hwaccel_device, "[1/6]", 1)
        self.assertIn("invalid device ordinal", detail)

        unwritable_output = [
            "[image2 @ 000] Could not open file : Z:/nope/out_0000_00.jpg",
            "[image2 @ 000] Could not write header (incorrect codec parameters ?): Input/output error",
            "Error initializing output stream: Error while opening output file",
            "Conversion failed!",
        ]
        detail = select_ffmpeg_error_detail(unwritable_output, "[1/6]", 1)
        self.assertIn("Could not open file", detail)
        # 無情報な要約だけが残ることは無い。
        self.assertNotEqual(detail, "Conversion failed!")

    def test_select_ffmpeg_error_detail_handles_missing_and_unmatched_output(self) -> None:
        self.assertEqual(
            select_ffmpeg_error_detail([], "[3/8]", 69),
            "[3/8] ffmpeg exited with code 69",
        )
        self.assertEqual(
            select_ffmpeg_error_detail(["   ", ""], "[3/8]", 1),
            "[3/8] ffmpeg exited with code 1",
        )
        # 該当行が無いときは末尾数行をそのまま返す。
        stats_only = ["frame=  10 fps=2.0 q=2.0 size=N/A time=00:00:05.00 bitrate=N/A speed=1.2x"]
        self.assertEqual(select_ffmpeg_error_detail(stats_only, "[1/1]", 1), stats_only[0])

    def test_select_ffmpeg_error_detail_is_bounded_and_deduplicated(self) -> None:
        repeated = ["[image2 @ 000] Could not open file : x.jpg"] * 10
        detail = select_ffmpeg_error_detail(repeated, "[1/1]", 1)
        self.assertEqual(detail, repeated[0])

        long_lines = [f"Error {i}: " + "x" * 400 for i in range(5)]
        detail = select_ffmpeg_error_detail(long_lines, "[1/1]", 1)
        self.assertLessEqual(len(detail), 500)
        self.assertTrue(detail.endswith("..."))

    def test_load_image_bgr_handles_non_ascii_paths_and_rejects_truncation(self) -> None:
        import numpy as np
        from PIL import Image

        with tempfile.TemporaryDirectory() as temp_dir:
            # cv2.imread はこのパスで None を返す。imdecode + np.fromfile なら読める。
            japanese_dir = Path(temp_dir) / "ドキュメント_出力" / "テスト フォルダ"
            japanese_dir.mkdir(parents=True)
            image_path = japanese_dir / "動画_0000_00.jpg"
            Image.fromarray(np.full((64, 48, 3), 120, dtype=np.uint8)).save(image_path, quality=95)

            loaded = load_image_bgr(image_path)
            self.assertEqual(np.asarray(loaded).shape, (64, 48, 3))

            raw = image_path.read_bytes()
            for fraction in (0.05, 0.5, 0.9, 0.999):
                truncated = japanese_dir / f"trunc_{fraction}.jpg"
                truncated.write_bytes(raw[: max(4, int(len(raw) * fraction))])
                with self.subTest(fraction=fraction):
                    # 途中まで書かれた JPEG を黙って読むとマスクが静かに壊れる。
                    with self.assertRaises(RuntimeError):
                        load_image_bgr(truncated)

            empty = japanese_dir / "empty.jpg"
            empty.write_bytes(b"")
            with self.assertRaises(RuntimeError):
                load_image_bgr(empty)

    def test_preload_native_backends_is_safe_to_call(self) -> None:
        # Tk を作る前に torch の DLL を読ませるためのフック。
        # 呼んでも例外を出さず、何度呼んでも良いこと。
        preload_native_backends()
        preload_native_backends()

    def test_main_preloads_before_creating_the_tk_root(self) -> None:
        # 順序が逆になると Windows でマスク生成がプロセスごと落ちる。
        import inspect
        import insta360_frame_extractor_gui as module

        source = inspect.getsource(module.main)
        self.assertLess(
            source.index("preload_native_backends()"),
            source.index("tk.Tk()"),
            "preload_native_backends() must run before tk.Tk()",
        )

    def test_mask_batch_limit_allows_one_pass_per_image(self) -> None:
        # 最高 は 2133x2133 で 9 タイルなので、上限が 9 未満だと必ず分割される。
        tiles = len(build_overlapping_tile_regions(2133, 2133, 1024, 256))
        self.assertEqual(tiles, 9)
        self.assertGreaterEqual(MASK_BATCH_SIZE_LIMIT, tiles)
        self.assertEqual(DEFAULT_MASK_BATCH_SIZE, tiles)
        # VRAM 実測 (B=32 で 4851 MiB) から 18 を超えないこと。
        self.assertLessEqual(MASK_BATCH_SIZE_LIMIT, 18)

    def test_parse_probability_threshold_rejects_zero_and_out_of_range(self) -> None:
        # 0 を許すと「確率 >= 0」が常に真になり全面黒マスクになる。
        for invalid_value in ("0", "0.0", "-0.1", "1.5", "", "abc"):
            with self.subTest(value=invalid_value):
                with self.assertRaises(ValueError):
                    parse_probability_threshold(invalid_value, "除外しきい値")

        self.assertEqual(parse_probability_threshold("0.7", "除外しきい値"), 0.7)
        self.assertEqual(parse_probability_threshold("1", "除外しきい値"), 1.0)

    def test_build_extracted_image_glob_escapes_special_characters(self) -> None:
        self.assertEqual(build_extracted_image_glob("clip"), "clip_*.jpg")
        self.assertEqual(build_extracted_image_glob(None), "*.jpg")
        self.assertEqual(build_extracted_image_glob(""), "*.jpg")
        # `[` をそのまま渡すと文字クラス扱いになりマッチしなくなる。
        self.assertEqual(build_extracted_image_glob("clip[1]"), "clip[[]1]_*.jpg")
        self.assertEqual(build_extracted_image_glob("clip*?"), "clip[*][?]_*.jpg")

    def test_extracted_image_glob_matches_bracketed_stem(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "clip[1]_0000_00.jpg").write_bytes(b"")
            (output_dir / "other_0000_00.jpg").write_bytes(b"")

            matched = sorted(path.name for path in output_dir.glob(build_extracted_image_glob("clip[1]")))
            self.assertEqual(matched, ["clip[1]_0000_00.jpg"])

    def test_build_tile_starts_covers_image_end(self) -> None:
        self.assertEqual(build_tile_starts(2560, 1024, 192), [0, 832, 1536])

    def test_build_overlapping_tile_regions_covers_corners(self) -> None:
        regions = build_overlapping_tile_regions(2560, 2560, 1024, 192)
        self.assertEqual(regions[0], (0, 0, 1024, 1024))
        self.assertEqual(regions[-1], (1536, 1536, 2560, 2560))

    def merged_command(self, directions=None, **kwargs) -> list[str]:
        options = dict(
            ffmpeg_path="ffmpeg",
            input_video=Path("C:/input/video.mp4"),
            output_dir=Path("C:/output"),
            video_stem="video",
            directions=directions if directions is not None else [Direction(yaw=45.0, pitch=-10.0)],
            fps=2.0,
            fov=100.0,
            width=1600,
            height=900,
        )
        options.update(kwargs)
        return build_merged_ffmpeg_command(**options)

    @staticmethod
    def output_groups(command: list[str]) -> list[tuple[str, list[str]]]:
        """(map ラベル, その出力の引数) に切り分ける。index 決め打ちの assert を避ける。"""
        groups: list[tuple[str, list[str]]] = []
        index = 0
        while index < len(command):
            if command[index] == "-map":
                label = command[index + 1]
                args: list[str] = []
                index += 2
                while index < len(command) and command[index] != "-map":
                    args.append(command[index])
                    index += 1
                groups.append((label, args))
            else:
                index += 1
        return groups

    @staticmethod
    def graph_of(command: list[str]) -> str:
        return command[command.index("-filter_complex") + 1]

    def test_shared_decode_head_orders_stages_by_cost(self) -> None:
        self.assertEqual(
            build_shared_decode_head(2, 0.5, use_gpu_decode=False, reverse_video=False),
            "[0:v]fps=0.5,split=2[s0][s1]",
        )
        self.assertEqual(
            build_shared_decode_head(2, 0.5, use_gpu_decode=True, reverse_video=False),
            "[0:v]fps=0.5,hwdownload,format=nv12,split=2[s0][s1]",
        )
        self.assertEqual(
            build_shared_decode_head(1, 24.0, use_gpu_decode=False, reverse_video=True),
            "[0:v]fps=24,reverse,split=1[s0]",
        )
        self.assertEqual(
            build_shared_decode_head(3, 1.5, use_gpu_decode=True, reverse_video=True),
            "[0:v]fps=1.5,hwdownload,format=nv12,reverse,split=3[s0][s1][s2]",
        )

    def test_direction_filter_graph_reports_its_output_pads(self) -> None:
        graph, outputs = build_direction_filter_graph(
            generate_ring_directions(4, 0.0), 1.0, 90.0, 1024, 1024
        )
        self.assertEqual(outputs, [("[e0]", 0), ("[e1]", 1), ("[e2]", 2), ("[e3]", 3)])
        self.assertEqual(graph.count(";"), 4)

        _, parity_outputs = build_direction_filter_graph(
            generate_ring_directions(4, 0.0), 1.0, 90.0, 1024, 1024,
            reverse_direction_index_on_odd=True,
        )
        self.assertEqual(
            parity_outputs,
            [("[e0]", 0), ("[o0]", 3), ("[e1]", 1), ("[o1]", 2),
             ("[e2]", 2), ("[o2]", 1), ("[e3]", 3), ("[o3]", 0)],
        )

    def test_merged_command_decodes_once_and_fans_out_to_every_direction(self) -> None:
        directions = generate_ring_directions(8, 0.0)
        command = self.merged_command(directions=directions)
        graph = self.graph_of(command)

        # デコード段は 1 本だけ。方向ごとに立てていた頃は N 回フルデコードしていた。
        self.assertEqual(graph.count("[0:v]"), 1)
        self.assertEqual(graph.count("split=8"), 1)
        self.assertEqual(graph.count("v360=input=equirect:output=flat"), 8)
        self.assertEqual(graph.count("interp=cubic"), 8)
        self.assertEqual(graph.count("setsar=1"), 8)
        self.assertIn("h_fov=100", graph)
        self.assertIn("v_fov=67.672748", graph)

        groups = self.output_groups(command)
        self.assertEqual(len(groups), 8)
        self.assertEqual([label for label, _ in groups], [f"[e{i}]" for i in range(8)])
        self.assertEqual(
            [args[-1].replace("\\", "/") for _, args in groups],
            [f"C:/output/video_%04d_{i:02d}.jpg" for i in range(8)],
        )

    def test_merged_command_sets_the_options_every_output_needs(self) -> None:
        command = self.merged_command(directions=generate_ring_directions(3, 0.0))
        for label, args in self.output_groups(command):
            with self.subTest(label=label):
                # select を挟むと passthrough なしで同じ絵が複数ファイルに書かれる。
                self.assertIn("-fps_mode", args)
                self.assertEqual(args[args.index("-fps_mode") + 1], "passthrough")
                self.assertIn("-frame_pts", args)
                self.assertEqual(args[args.index("-frame_pts") + 1], "1")
                self.assertIn("-atomic_writing", args)
                self.assertEqual(args[args.index("-atomic_writing") + 1], "1")
                self.assertEqual(args[args.index("-q:v") + 1], "2")
                self.assertEqual(args[args.index("-threads:v") + 1], "1")
        # 一時ディレクトリ経由をやめたので -start_number は使わない。
        self.assertNotIn("-start_number", command)

    def test_merged_command_orders_the_shared_head_for_memory_safety(self) -> None:
        graph = self.graph_of(
            self.merged_command(
                directions=generate_ring_directions(4, 0.0),
                use_gpu_decode=True,
                reverse_video=True,
            )
        )
        head = graph.split(";")[0]
        # fps -> hwdownload,format -> reverse -> split の順序が崩れると
        # 転送量が 48 倍になったり VRAM / ホスト RAM が飛ぶ。
        self.assertEqual(
            head,
            "[0:v]fps=2,hwdownload,format=nv12,reverse,split=4[s0][s1][s2][s3]",
        )

    def test_merged_command_adds_cuda_decode_when_requested(self) -> None:
        command = self.merged_command(use_gpu_decode=True)
        self.assertEqual(command[3:7], ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"])
        self.assertIn("hwdownload,format=nv12", self.graph_of(command))

        cpu_command = self.merged_command(use_gpu_decode=False)
        self.assertNotIn("-hwaccel", cpu_command)
        self.assertNotIn("hwdownload", self.graph_of(cpu_command))

    def test_merged_command_drops_frames_before_downloading_them(self) -> None:
        graph = self.graph_of(self.merged_command(use_gpu_decode=True))
        self.assertLess(graph.index("fps=2"), graph.index("hwdownload"))

    def test_merged_command_splits_parities_into_their_final_suffixes(self) -> None:
        directions = generate_ring_directions(4, 0.0)
        command = self.merged_command(
            directions=directions,
            reverse_direction_index_on_odd=True,
        )
        graph = self.graph_of(command)
        # カンマを含む式は filter_complex でエスケープが要るので使わない。
        self.assertNotIn("mod(", graph)
        self.assertEqual(graph.count("select=not(n-2*trunc(n/2))"), 4)
        self.assertEqual(graph.count("select=n-2*trunc(n/2)"), 4)

        groups = self.output_groups(command)
        self.assertEqual(len(groups), 8)
        produced = {}
        for label, args in groups:
            produced[label] = args[-1].replace("\\", "/")
        for index in range(4):
            even_expected = build_output_image_path(
                Path("C:/output"), "video", 0, index, 4, reverse_on_odd_frames=True
            )
            odd_expected = build_output_image_path(
                Path("C:/output"), "video", 1, index, 4, reverse_on_odd_frames=True
            )
            # ffmpeg が書くパターンは、旧実装のリネーム結果と同じサフィックスになる。
            self.assertTrue(produced[f"[e{index}]"].endswith(f"_{index:02d}.jpg"))
            self.assertEqual(even_expected.name.rsplit("_", 1)[1], f"{index:02d}.jpg")
            self.assertTrue(produced[f"[o{index}]"].endswith(f"_{3 - index:02d}.jpg"))
            self.assertEqual(odd_expected.name.rsplit("_", 1)[1], f"{3 - index:02d}.jpg")

    def test_merged_command_shares_one_pattern_for_the_odd_count_centre(self) -> None:
        # N が奇数だと中央方向は (N-1)-i == i なので両パリティが同じパターンを共有する。
        # フレーム番号の偶奇が互いに素なので衝突しない。
        command = self.merged_command(
            directions=generate_ring_directions(3, 0.0),
            reverse_direction_index_on_odd=True,
        )
        groups = self.output_groups(command)
        centre = [args[-1] for label, args in groups if label in ("[e1]", "[o1]")]
        self.assertEqual(len(centre), 2)
        self.assertEqual(centre[0], centre[1])

    def test_merged_command_without_parity_has_one_output_per_direction(self) -> None:
        command = self.merged_command(
            directions=generate_ring_directions(5, 0.0),
            reverse_direction_index_on_odd=False,
        )
        graph = self.graph_of(command)
        self.assertNotIn("select=", graph)
        self.assertEqual(len(self.output_groups(command)), 5)

    def test_merged_command_rejects_an_empty_direction_list(self) -> None:
        with self.assertRaises(ValueError):
            self.merged_command(directions=[])

    def test_merged_command_stays_within_the_windows_argument_limit(self) -> None:
        # 15 方向 + parity = 30 出力。これが実運用の最大構成。
        command = self.merged_command(
            directions=generate_ring_directions(15, 10.0),
            reverse_direction_index_on_odd=True,
            use_gpu_decode=True,
        )
        self.assertEqual(len(self.output_groups(command)), 30)
        self.assertLess(sum(len(argument) + 1 for argument in command), 30000)

    def test_compute_expected_frame_count_matches_the_fps_filter(self) -> None:
        # fps フィルタは t = k/fps < duration の k を出すので ceil(duration*fps) 枚。
        # 実測: 561.791667 s / fps 1.5 -> 843 枚、40.0 s / fps 1.5 -> 60 枚。
        self.assertEqual(compute_expected_frame_count(561.791667, 1.5), 843)
        self.assertEqual(compute_expected_frame_count(561.791667, 0.5), 281)
        # duration*fps がちょうど整数のときに 1 枚多く数えないこと。
        self.assertEqual(compute_expected_frame_count(40.0, 1.5), 60)
        self.assertEqual(compute_expected_frame_count(10.0, 1.0), 10)
        self.assertEqual(compute_expected_frame_count(0.5, 1.0), 1)
        self.assertIsNone(compute_expected_frame_count(None, 1.0))
        self.assertIsNone(compute_expected_frame_count(0.0, 1.0))
        self.assertIsNone(compute_expected_frame_count(10.0, 0.0))

    def test_parse_ffmpeg_time_seconds_reads_the_stats_line(self) -> None:
        line = "frame=  10 fps=2.0 q=2.0 size=N/A time=00:01:23.45 bitrate=N/A speed=1.2x"
        self.assertAlmostEqual(parse_ffmpeg_time_seconds(line), 83.45, places=3)
        self.assertAlmostEqual(parse_ffmpeg_time_seconds("time=01:00:00.00"), 3600.0, places=3)
        # 最初の統計行は time=N/A になることがある。
        self.assertIsNone(parse_ffmpeg_time_seconds("frame=0 fps=0.0 time=N/A"))
        self.assertIsNone(parse_ffmpeg_time_seconds("Stream #0:0 -> #0:0 (hevc -> mjpeg)"))

    def test_is_ffmpeg_progress_line_only_matches_stats(self) -> None:
        self.assertTrue(is_ffmpeg_progress_line("frame=  10 fps=2.0 time=00:00:05.00"))
        self.assertTrue(is_ffmpeg_progress_line("size=N/A time=00:00:05.00"))
        self.assertFalse(is_ffmpeg_progress_line("[image2 @ 0] Could not open file"))
        self.assertFalse(is_ffmpeg_progress_line("Conversion failed!"))

    def test_generate_ring_directions_evenly_spreads_yaw(self) -> None:
        directions = generate_ring_directions(4, 15.0)
        self.assertEqual(
            [(item.yaw, item.pitch) for item in directions],
            [(0.0, 15.0), (90.0, 15.0), (180.0, 15.0), (-90.0, 15.0)],
        )

    def test_parse_angle_normalizes_large_values(self) -> None:
        self.assertEqual(parse_angle("270", "yaw"), -90.0)
        self.assertEqual(parse_angle("540", "yaw"), 180.0)

    def test_compute_vertical_fov_matches_output_aspect_ratio(self) -> None:
        self.assertAlmostEqual(compute_vertical_fov(100.0, 1600, 900), 67.672748, places=5)

    def test_direction_to_equirect_point_maps_forward_to_center(self) -> None:
        point = direction_to_equirect_point(Direction(yaw=0.0, pitch=0.0), 4000, 2000)
        self.assertEqual(point, (2000.0, 1000.0))

    def test_compute_preview_box_preserves_aspect_ratio(self) -> None:
        preview_box = compute_preview_box(5760, 2880, 800, 400)
        self.assertEqual(preview_box, (0.0, 0.0, 800.0, 400.0))

    def test_fit_size_within_bounds_scales_to_available_space(self) -> None:
        fitted = fit_size_within_bounds(7680, 3840, 1400, 700)
        self.assertEqual(fitted, (1400, 700))

    def test_fit_size_within_bounds_never_exceeds_bounds(self) -> None:
        # 極端に小さい / 壊れた領域でも枠外に出るサイズを返さない。
        for max_width, max_height in ((1, 1), (10, 4), (0, 0), (-5, -5)):
            with self.subTest(bounds=(max_width, max_height)):
                width, height = fit_size_within_bounds(5760, 2880, max_width, max_height, allow_upscale=True)
                self.assertGreaterEqual(width, 1)
                self.assertGreaterEqual(height, 1)
                self.assertLessEqual(width, max(1, max_width))
                self.assertLessEqual(height, max(1, max_height))

    def test_geometry_helpers_survive_degenerate_sizes(self) -> None:
        # 0 除算せず、拡大が許可されていなければ元サイズ以下に収める。
        self.assertEqual(fit_size_within_bounds(0, 0, 100, 100), (1, 1))
        self.assertEqual(fit_size_within_bounds(0, 0, 100, 100, allow_upscale=True), (100, 100))
        offset_x, offset_y, display_width, display_height = compute_preview_box(0, 0, 0, 0)
        self.assertEqual((offset_x, offset_y), (0.0, 0.0))
        self.assertEqual((display_width, display_height), (1.0, 1.0))

    def test_footprint_segments_stay_in_map_space(self) -> None:
        segments = build_footprint_segments(
            direction=Direction(yaw=45.0, pitch=10.0),
            horizontal_fov=90.0,
            aspect_ratio=16 / 9,
            map_width=4000,
            map_height=2000,
        )

        self.assertGreater(len(segments), 0)
        flattened = [point for segment in segments for point in segment]
        self.assertTrue(all(0.0 <= point[1] <= 2000.0 for point in flattened))

    def test_yaw_180_footprint_keeps_single_unwrapped_segment(self) -> None:
        segments = build_footprint_segments(
            direction=Direction(yaw=180.0, pitch=0.0),
            horizontal_fov=90.0,
            aspect_ratio=16 / 9,
            map_width=4000,
            map_height=2000,
        )

        self.assertEqual(len(segments), 1)
        self.assertGreater(len(segments[0]), 20)


def build_headless_app(settings_path: Path):
    """設定ファイルを差し替えた上で GUI を非表示のまま組み立てる。"""
    import insta360_frame_extractor_gui as app_module

    original_settings_path = app_module.default_settings_path
    app_module.default_settings_path = lambda: settings_path
    try:
        root = tkinter.Tk()
        root.withdraw()
        app = app_module.Insta360ExtractorApp(root)
        root.withdraw()
        # 破棄後に after コールバックが走って Tcl エラーになるのを防ぐ。
        app._cancel_pending_callbacks()
        return app, root
    finally:
        app_module.default_settings_path = original_settings_path


def tk_is_available() -> bool:
    try:
        probe = tkinter.Tk()
    except Exception:
        return False
    probe.destroy()
    return True


@unittest.skipUnless(tk_is_available(), "Tk が利用できない環境ではスキップします。")
class DirectionSetStateTests(unittest.TestCase):
    """`self.directions` が常にアクティブな DirectionSet と同じリストを指すことの回帰テスト。"""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.settings_path = Path(self.temp_dir.name) / ".settings.json"
        self.app, self.root = build_headless_app(self.settings_path)
        self.addCleanup(self.temp_dir.cleanup)
        self.addCleanup(self.root.destroy)

    def active_directions(self) -> list[Direction]:
        return self.app.direction_sets[self.app.active_direction_set_index].directions

    def test_replace_with_ring_updates_active_direction_set(self) -> None:
        self.app.ring_count_var.set("4")
        self.app.ring_pitch_var.set("0")
        self.app._replace_with_ring()

        self.assertIs(self.app.directions, self.active_directions())
        self.assertEqual(len(self.active_directions()), 4)

    def test_clear_directions_updates_active_direction_set(self) -> None:
        self.app.ring_count_var.set("5")
        self.app._replace_with_ring()
        self.app._clear_directions()

        self.assertIs(self.app.directions, self.active_directions())
        self.assertEqual(self.active_directions(), [Direction(yaw=0.0, pitch=0.0)])

    def test_removing_last_direction_keeps_active_direction_set_bound(self) -> None:
        self.app.ring_count_var.set("1")
        self.app._replace_with_ring()
        self.app._refresh_direction_table(select_index=0)
        self.app._remove_selected_direction()

        self.assertIs(self.app.directions, self.active_directions())
        self.assertEqual(len(self.active_directions()), 1)

    def test_switching_direction_sets_preserves_each_edit(self) -> None:
        self.app.direction_sets.append(DirectionSet(name="セット2", directions=[Direction(yaw=10.0, pitch=5.0)]))
        self.app._refresh_direction_tabs()

        self.app.ring_count_var.set("3")
        self.app._replace_with_ring()
        ring_directions = list(self.app.directions)

        self.app._switch_direction_set(1)
        self.assertEqual(self.app.directions, [Direction(yaw=10.0, pitch=5.0)])

        self.app._switch_direction_set(0)
        self.assertEqual(self.app.directions, ring_directions)

    def test_ring_survives_settings_round_trip(self) -> None:
        self.app.ring_count_var.set("6")
        self.app._replace_with_ring()
        expected = list(self.app.directions)
        self.app._save_persisted_settings()

        reloaded_app, reloaded_root = build_headless_app(self.settings_path)
        self.addCleanup(reloaded_root.destroy)
        self.assertEqual(reloaded_app.directions, expected)


@unittest.skipUnless(tk_is_available(), "Tk が利用できない環境ではスキップします。")
class PreviewLayoutTests(unittest.TestCase):
    """どのウィンドウサイズでもプレビュー面が表示領域を超えないことを確認する。"""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.app, self.root = build_headless_app(Path(self.temp_dir.name) / ".settings.json")
        self.addCleanup(self.temp_dir.cleanup)
        self.addCleanup(self.root.destroy)

    def test_layout_fits_inside_media_frame_for_any_window_size(self) -> None:
        from insta360_frame_extractor_gui import VideoMetadata

        self.root.deiconify()
        self.root.update()
        for metadata in (
            None,
            VideoMetadata(width=5760, height=2880, duration_seconds=12.0),
            VideoMetadata(width=1, height=1, duration_seconds=None),
            VideoMetadata(width=1080, height=1920, duration_seconds=3.0),
        ):
            for window_width, window_height in ((640, 480), (1180, 800), (1920, 1200), (400, 300)):
                with self.subTest(metadata=metadata, size=(window_width, window_height)):
                    self.app.preview_metadata = metadata
                    self.root.geometry(f"{window_width}x{window_height}")
                    self.root.update()
                    self.app._layout_preview_surface()
                    self.root.update()

                    frame_width = self.app.preview_media_frame.winfo_width()
                    frame_height = self.app.preview_media_frame.winfo_height()
                    self.assertGreaterEqual(self.app.preview_canvas_width, 1)
                    self.assertGreaterEqual(self.app.preview_canvas_height, 1)
                    self.assertLessEqual(
                        self.app.preview_canvas_offset_x + self.app.preview_canvas_width,
                        frame_width,
                    )
                    self.assertLessEqual(
                        self.app.preview_canvas_offset_y + self.app.preview_canvas_height,
                        frame_height,
                    )

    def test_minimum_window_size_shows_every_widget(self) -> None:
        self.root.deiconify()
        self.root.update_idletasks()
        min_width, min_height = self.root.minsize()
        self.root.geometry(f"{min_width}x{min_height}")
        self.root.update_idletasks()

        # 最小サイズでも要求サイズを満たしていれば、どのウィジェットも潰れない。
        self.assertGreaterEqual(min_width, self.root.winfo_reqwidth())
        self.assertGreaterEqual(min_height, self.root.winfo_reqheight())

    def test_preview_mode_switches_back_from_vlc(self) -> None:
        # 高速再生を使った後に「枠位置優先」へ戻せることを保証する。
        self.app.preview_mode = "vlc"
        self.app.preview_vlc_player = object()
        self.assertTrue(self.app._preview_uses_vlc())

        self.app._close_preview_capture()
        self.assertEqual(self.app.preview_mode, "none")
        self.assertFalse(self.app._preview_uses_vlc())


if __name__ == "__main__":
    unittest.main()
