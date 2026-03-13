#!/usr/bin/env python3
"""
RealSense Camera RGB Viewer Test

Display live RGB video stream from a RealSense camera.

Usage:
    python realsense_camera_test.py --camera-serial 825412070292
    python realsense_camera_test.py --camera-serial 825412070487
    python realsense_camera_test.py --camera-serial 825412070487 --show-depth
    python realsense_camera_test.py  # Use default camera
    uv run realsense_camera_test.py --external-serial 825412070487 --wrist-serial 825412070292

Press 'q' to quit.
"""

import argparse
import cv2
import numpy as np

from control.camera_connector import RealSenseConnector


def main():
    parser = argparse.ArgumentParser(description="RealSense Camera RGB Viewer")
    parser.add_argument(
        "--camera-serial",
        type=str,
        default=None,
        help="RealSense camera serial number (e.g., 825412070292)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Camera resolution width",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Camera resolution height",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Camera FPS",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=1000,
        help="Frame capture timeout in milliseconds",
    )
    parser.add_argument(
        "--show-depth",
        action="store_true",
        help="Also show depth view (side by side with RGB)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("RealSense Camera RGB Viewer")
    print("=" * 70)
    print(f"Camera serial: {args.camera_serial or 'default'}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"FPS: {args.fps}")
    print(f"Show depth: {args.show_depth}")
    print("\nPress 'q' to quit")
    print("=" * 70)

    # Initialize camera
    camera = RealSenseConnector(
        serial=args.camera_serial,
        width=args.width,
        height=args.height,
        fps=args.fps,
        enable_depth=args.show_depth,
        align_depth_to_color=True,
    )

    window_name = f"RealSense Camera - {args.camera_serial or 'default'}"

    try:
        camera.connect()
        print(f"\nConnected to camera!")
        print(f"Intrinsics K:\n{camera.intrinsics}\n")

        frame_count = 0
        while True:
            # Update camera frame
            ok = camera.update(timeout=args.timeout_ms)
            if not ok:
                print(f"[{frame_count}] Failed to get frame")
                continue

            rgb = camera.img
            if rgb is None:
                print(f"[{frame_count}] No RGB image")
                continue

            # Convert RGB to BGR for OpenCV display
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # Show depth alongside RGB if requested
            if args.show_depth:
                depth = camera.depth
                if depth is not None:
                    # Normalize depth for visualization
                    depth_normalized = np.clip(depth * 1000, 0, 2000)  # 0-2m range
                    depth_vis = (depth_normalized / 2000 * 255).astype(np.uint8)
                    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

                    # Combine RGB and depth side by side
                    combined = np.hstack([bgr, depth_colored])
                    display_img = combined
                else:
                    display_img = bgr
            else:
                display_img = bgr

            # Add frame counter
            cv2.putText(
                display_img,
                f"Frame: {frame_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            # Display
            cv2.imshow(window_name, display_img)

            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\n'q' pressed, exiting...")
                break

            frame_count += 1

    except KeyboardInterrupt:
        print("\nCtrl+C detected, exiting...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
    finally:
        camera.close()
        cv2.destroyAllWindows()
        print("Camera closed")


if __name__ == "__main__":
    main()
