import unittest
from pathlib import Path

from insta360_frame_extractor_gui import (
    Direction,
    build_footprint_segments,
    build_ffmpeg_command,
    build_mask_output_path,
    build_overlapping_tile_regions,
    build_output_image_path,
    build_output_pattern,
    build_tile_starts,
    build_binary_usage_mask,
    collect_segformer_excluded_label_groups,
    collect_segformer_excluded_label_ids,
    compute_output_direction_index,
    compute_vertical_fov,
    compute_preview_box,
    direction_to_equirect_point,
    direction_index_width,
    fit_size_within_bounds,
    generate_ring_directions,
    mask_contains_excluded_region,
    parse_angle,
    resolve_mask_detail_preset,
    resize_label_prediction_nearest,
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

    def test_collect_segformer_excluded_label_ids_finds_sky_person_car(self) -> None:
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
        self.assertEqual(label_ids, {1, 2, 4})

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

    def test_resize_label_prediction_nearest_preserves_labels(self) -> None:
        resized = resize_label_prediction_nearest(
            [[1, 2], [3, 4]],
            (4, 4),
        )
        self.assertEqual(resized.shape, (4, 4))
        self.assertEqual(resized[0, 0], 1)
        self.assertEqual(resized[0, 3], 2)
        self.assertEqual(resized[3, 0], 3)
        self.assertEqual(resized[3, 3], 4)

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


if __name__ == "__main__":
    unittest.main()
