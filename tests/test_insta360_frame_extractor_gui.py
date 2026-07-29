import math
import tempfile
import tkinter
import unittest
from pathlib import Path
from xml.etree import ElementTree

from insta360_frame_extractor_gui import (
    Direction,
    DirectionSet,
    build_extracted_image_glob,
    build_filter,
    build_footprint_segments,
    build_ffmpeg_command,
    build_mask_output_path,
    build_overlapping_tile_regions,
    build_output_image_path,
    build_output_pattern,
    build_single_pass_ffmpeg_command,
    build_single_pass_filter_complex,
    build_tile_starts,
    build_xmp_document,
    build_xmp_sidecar_path,
    build_binary_usage_mask,
    collect_segformer_excluded_label_groups,
    collect_segformer_excluded_label_ids,
    compute_focal_length_35mm,
    compute_output_direction_index,
    compute_vertical_fov,
    compute_preview_box,
    direction_to_equirect_point,
    direction_index_width,
    fit_size_within_bounds,
    generate_ring_directions,
    mask_contains_excluded_region,
    parse_angle,
    parse_probability_threshold,
    resolve_mask_detail_preset,
)


class SinglePassExtractionTests(unittest.TestCase):
    directions = [Direction(yaw=0.0, pitch=0.0), Direction(yaw=90.0, pitch=-10.0)]

    def test_single_pass_decodes_input_once_for_every_direction(self) -> None:
        command = build_single_pass_ffmpeg_command(
            ffmpeg_path="ffmpeg",
            input_video=Path("C:/in/clip.mp4"),
            output_patterns=[Path("C:/tmp/0/%04d.jpg"), Path("C:/tmp/1/%04d.jpg")],
            directions=self.directions,
            fps=1.0,
            fov=90.0,
            width=1920,
            height=1080,
        )
        self.assertEqual(command.count("-i"), 1)
        self.assertEqual(command.count("-map"), len(self.directions))
        self.assertEqual(command.count("-filter_complex"), 1)

    def test_single_pass_filter_complex_splits_after_the_shared_fps_stage(self) -> None:
        filter_complex, labels = build_single_pass_filter_complex(
            self.directions,
            fps=2.0,
            fov=90.0,
            width=1920,
            height=1440,
            use_gpu_decode=False,
        )
        self.assertEqual(labels, ["o0", "o1"])
        # fps は分岐前に 1 度だけ適用され、以降は方向ごとの v360 だけが走る。
        self.assertEqual(filter_complex.count("fps=2"), 1)
        self.assertEqual(filter_complex.count("v360="), len(self.directions))
        self.assertIn("split=2[s0][s1]", filter_complex)

    def test_single_pass_branches_match_the_per_direction_filter(self) -> None:
        # 1 パス化しても各方向に適用される変換は従来と同一でなければならない。
        filter_complex, _ = build_single_pass_filter_complex(
            self.directions,
            fps=1.0,
            fov=90.0,
            width=1920,
            height=1080,
            use_gpu_decode=False,
        )
        for index, direction in enumerate(self.directions):
            reference = build_filter(direction, 1.0, 90.0, 1920, 1080)
            v360_stage = reference.split(",", 1)[1]
            self.assertIn(f"[s{index}]{v360_stage}[o{index}]", filter_complex)

    def test_single_pass_downloads_gpu_frames_before_splitting(self) -> None:
        filter_complex, _ = build_single_pass_filter_complex(
            self.directions,
            fps=1.0,
            fov=90.0,
            width=1920,
            height=1080,
            use_gpu_decode=True,
        )
        self.assertTrue(filter_complex.startswith("[0:v]hwdownload,format=nv12,fps="))

    def test_single_pass_rejects_mismatched_output_patterns(self) -> None:
        with self.assertRaises(ValueError):
            build_single_pass_ffmpeg_command(
                ffmpeg_path="ffmpeg",
                input_video=Path("C:/in/clip.mp4"),
                output_patterns=[Path("C:/tmp/0/%04d.jpg")],
                directions=self.directions,
                fps=1.0,
                fov=90.0,
                width=1920,
                height=1080,
            )


class RealityScanXmpTests(unittest.TestCase):
    def test_focal_length_35mm_matches_the_requested_horizontal_fov(self) -> None:
        # h_fov 90 度なら f = (w/2)/tan(45) = w/2 → 長辺基準で 36 * 0.5 = 18mm。
        self.assertAlmostEqual(compute_focal_length_35mm(90.0, 1920, 1080), 18.0, places=9)
        # 画角を半分にすると焦点距離はほぼ倍になる。
        self.assertAlmostEqual(compute_focal_length_35mm(45.0, 1920, 1080), 18.0 / math.tan(math.radians(22.5)), places=9)

    def test_focal_length_35mm_normalises_by_the_longer_edge(self) -> None:
        landscape = compute_focal_length_35mm(90.0, 1920, 1080)
        square = compute_focal_length_35mm(90.0, 1080, 1080)
        self.assertAlmostEqual(landscape, square, places=9)

    def test_sidecar_path_replaces_the_image_extension(self) -> None:
        sidecar = build_xmp_sidecar_path(Path("C:/output/clip_0000_00.jpg"))
        self.assertEqual(str(sidecar).replace("\\", "/"), "C:/output/clip_0000_00.xmp")

    def test_xmp_document_declares_the_capturing_reality_namespace(self) -> None:
        document = build_xmp_document(18.0)
        self.assertIn('xmlns:xcr="http://www.capturingreality.com/ns/xcr/1.1#"', document)
        self.assertIn('xcr:FocalLength35mm="18.000000000"', document)
        self.assertIn('xcr:DistortionModel="division"', document)

    def test_xmp_document_shares_one_calibration_group_across_images(self) -> None:
        # 全画像が同一の合成カメラなので、キャリブレーションは 1 グループにまとめる。
        first = build_xmp_document(18.0, calibration_group=0, distortion_group=0)
        second = build_xmp_document(18.0, calibration_group=0, distortion_group=0)
        self.assertEqual(first, second)
        self.assertIn('xcr:CalibrationGroup="0"', first)
        self.assertIn('xcr:DistortionGroup="0"', first)

    def test_xmp_document_omits_unverified_pose_priors(self) -> None:
        # 回転行列の座標系規約を実機確認できていないため、姿勢は書き出さない。
        document = build_xmp_document(18.0)
        self.assertNotIn("xcr:Rotation", document)
        self.assertNotIn("xcr:Position", document)

    def test_xmp_document_is_well_formed_xml(self) -> None:
        ElementTree.fromstring(build_xmp_document(18.0))


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

    def test_build_ffmpeg_command_uses_v360_and_jpg_sequence(self) -> None:
        command = build_ffmpeg_command(
            ffmpeg_path="ffmpeg",
            input_video=Path("C:/input/video.mp4"),
            output_dir=Path("C:/output"),
            video_stem="video",
            direction=Direction(yaw=45.0, pitch=-10.0),
            direction_idx=3,
            direction_count=8,
            fps=2.0,
            fov=100.0,
            width=1600,
            height=900,
        )

        self.assertIn("fps=2", command[7])
        self.assertIn("v360=input=equirect:output=flat", command[7])
        self.assertIn("interp=cubic", command[7])
        self.assertIn("h_fov=100", command[7])
        self.assertIn("v_fov=67.672748", command[7])
        self.assertIn("setsar=1", command[7])
        self.assertIn("yaw=45", command[7])
        self.assertIn("pitch=-10", command[7])
        self.assertEqual(command[-1].replace("\\", "/"), "C:/output/video_%04d_03.jpg")

    def test_build_ffmpeg_command_adds_cuda_decode_when_requested(self) -> None:
        command = build_ffmpeg_command(
            ffmpeg_path="ffmpeg",
            input_video=Path("C:/input/video.mp4"),
            output_dir=Path("C:/output"),
            video_stem="video",
            direction=Direction(yaw=45.0, pitch=-10.0),
            direction_idx=3,
            direction_count=8,
            fps=2.0,
            fov=100.0,
            width=1600,
            height=900,
            use_gpu_decode=True,
        )

        self.assertEqual(command[3:7], ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"])
        self.assertIn("hwdownload,format=nv12", command[11])

    def test_build_ffmpeg_command_adds_reverse_when_requested(self) -> None:
        command = build_ffmpeg_command(
            ffmpeg_path="ffmpeg",
            input_video=Path("C:/input/video.mp4"),
            output_dir=Path("C:/output"),
            video_stem="video",
            direction=Direction(yaw=45.0, pitch=-10.0),
            direction_idx=3,
            direction_count=8,
            fps=2.0,
            fov=100.0,
            width=1600,
            height=900,
            reverse_video=True,
        )

        self.assertIn("fps=2,reverse,", command[7])

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
