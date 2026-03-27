/**
 * @file capture_images.cpp
 * @brief Camera capture tool — save images for dataset collection or camera testing.
 *
 * Usage:
 *   capture_images                              # USB cam 0, interactive mode
 *   capture_images --camera 0 --interval 500    # auto-capture every 500ms
 *   capture_images --camera 0 --count 100       # capture exactly 100 images
 *   capture_images --output data/captures        # save to custom directory
 *
 * Modes:
 *   Interactive (default):  Press SPACE to capture, Q/ESC to quit
 *   Auto-interval:          --interval <ms>  captures automatically
 *   Fixed count:            --count <n>      stops after N captures
 *
 * Captured images are saved as JPEG with timestamps in the output directory.
 */

#include <atomic>
#include <chrono>
#include <csignal>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "edgeai/camera/camera_manager.h"
#include "edgeai/common/logger.h"
#include "edgeai/common/types.h"

namespace fs = std::filesystem;

static std::atomic<bool> g_running{true};

void signal_handler(int) {
    g_running = false;
}

struct CaptureArgs {
    int         camera_id     = 0;
    int         width         = 1920;
    int         height        = 1080;
    int         fps           = 30;
    std::string backend       = "v4l2";
    std::string output_dir    = "data/captures";
    std::string video_path;             // empty = live camera, set = read from file
    int         interval_ms   = 0;      // 0 = interactive mode
    int         count         = 0;      // 0 = unlimited
    int         jpeg_quality  = 95;
    bool        show_preview  = true;
    bool        help          = false;
};

CaptureArgs parse_args(int argc, char* argv[]) {
    CaptureArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--camera" || arg == "-c") && i + 1 < argc)
            args.camera_id = std::stoi(argv[++i]);
        else if (arg == "--width" && i + 1 < argc)
            args.width = std::stoi(argv[++i]);
        else if (arg == "--height" && i + 1 < argc)
            args.height = std::stoi(argv[++i]);
        else if (arg == "--fps" && i + 1 < argc)
            args.fps = std::stoi(argv[++i]);
        else if (arg == "--backend" && i + 1 < argc)
            args.backend = argv[++i];
        else if ((arg == "--output" || arg == "-o") && i + 1 < argc)
            args.output_dir = argv[++i];
        else if ((arg == "--video" || arg == "-v") && i + 1 < argc)
            args.video_path = argv[++i];
        else if ((arg == "--interval" || arg == "-i") && i + 1 < argc)
            args.interval_ms = std::stoi(argv[++i]);
        else if ((arg == "--count" || arg == "-n") && i + 1 < argc)
            args.count = std::stoi(argv[++i]);
        else if (arg == "--quality" && i + 1 < argc)
            args.jpeg_quality = std::stoi(argv[++i]);
        else if (arg == "--no-preview")
            args.show_preview = false;
        else if (arg == "--help" || arg == "-h")
            args.help = true;
    }
    return args;
}

void print_usage() {
    std::cout << R"(
EdgeAI Camera Capture Tool
===========================

Usage:
  capture_images [options]

Options:
  --camera, -c <id>      Camera device ID (default: 0)
  --video, -v <path>     Read frames from video file instead of camera
  --width <px>           Frame width (default: 1920)
  --height <px>          Frame height (default: 1080)
  --fps <n>              Frames per second (default: 30)
  --backend <type>       Camera backend: v4l2, gstreamer, usb3 (default: v4l2)
  --output, -o <dir>     Output directory (default: data/captures)
  --interval, -i <ms>    Auto-capture interval in milliseconds (0 = interactive)
  --count, -n <num>      Stop after N captures (0 = unlimited)
  --quality <1-100>      JPEG quality (default: 95)
  --no-preview           Disable live preview window
  --help, -h             Show this help

Modes:
  Interactive (default):  Press SPACE to save frame, Q/ESC to quit
  Auto-interval:          --interval 500  (captures every 500ms)
  Fixed count:            --count 100     (stops after 100 captures)

Examples:
  capture_images                                # interactive, USB cam 0
  capture_images --camera 0 --interval 1000     # auto-capture every 1 second
  capture_images -c 0 -n 200 -o dataset/train   # 200 images to dataset/train/
  capture_images --camera 0 --no-preview -i 500  # headless auto-capture
  capture_images --video test.mp4 -n 20           # extract 20 frames from video

)" << std::endl;
}

std::string make_filename(const std::string& dir, int index) {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&time, &tm);

    char timestamp[32];
    std::strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", &tm);

    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;

    return dir + "/img_" + std::string(timestamp)
        + "_" + std::to_string(ms.count())
        + "_" + std::to_string(index) + ".jpg";
}

int main(int argc, char* argv[]) {
    auto args = parse_args(argc, argv);
    if (args.help) {
        print_usage();
        return 0;
    }

    // Setup signal handler
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    const bool use_video = !args.video_path.empty();

    // --- Video-file source ---
    cv::VideoCapture video_cap;
    // --- Live-camera source ---
    std::unique_ptr<edgeai::CameraManager> camera;

    if (use_video) {
        if (!fs::exists(args.video_path)) {
            std::cerr << "ERROR: Video file not found: " << args.video_path << std::endl;
            return 1;
        }
        video_cap.open(args.video_path);
        if (!video_cap.isOpened()) {
            std::cerr << "ERROR: Failed to open video file: " << args.video_path << std::endl;
            return 1;
        }
        // Override resolution/fps from actual video metadata
        args.width  = static_cast<int>(video_cap.get(cv::CAP_PROP_FRAME_WIDTH));
        args.height = static_cast<int>(video_cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        args.fps    = static_cast<int>(video_cap.get(cv::CAP_PROP_FPS));
    } else {
        edgeai::CameraConfig cam_cfg;
        cam_cfg.device_id = args.camera_id;
        cam_cfg.width     = args.width;
        cam_cfg.height    = args.height;
        cam_cfg.fps       = args.fps;
        cam_cfg.backend   = args.backend;
        camera = std::make_unique<edgeai::CameraManager>(cam_cfg);
    }

    // Banner
    std::cout << "\n╔══════════════════════════════════════════╗" << std::endl;
    std::cout << "║    EdgeAI Camera Capture Tool            ║" << std::endl;
    std::cout << "╠══════════════════════════════════════════╣" << std::endl;
    if (use_video) {
        std::cout << "║ Source:     " << args.video_path << std::endl;
        int total_frames = static_cast<int>(video_cap.get(cv::CAP_PROP_FRAME_COUNT));
        std::cout << "║ Frames:     " << total_frames << std::endl;
    } else {
        std::cout << "║ Camera:     /dev/video" << args.camera_id << std::endl;
    }
    std::cout << "║ Resolution: " << args.width << "x" << args.height
              << " @ " << args.fps << " fps" << std::endl;
    std::cout << "║ Output:     " << args.output_dir << std::endl;
    std::cout << "║ Mode:       "
              << (args.interval_ms > 0
                  ? "Auto (" + std::to_string(args.interval_ms) + "ms interval)"
                  : (use_video ? "Every frame" : "Interactive (SPACE=capture, Q=quit)"))
              << std::endl;
    if (args.count > 0) {
        std::cout << "║ Count:      " << args.count << " images" << std::endl;
    }
    std::cout << "╚══════════════════════════════════════════╝\n" << std::endl;

    // Open camera (video already opened above)
    if (!use_video) {
        if (!camera->open()) {
            std::cerr << "ERROR: Failed to open camera /dev/video"
                      << args.camera_id << std::endl;
            std::cerr << "  Check: ls /dev/video*" << std::endl;
            std::cerr << "  Check: v4l2-ctl --list-devices" << std::endl;
            return 1;
        }
    }

    // Create output directory
    fs::create_directories(args.output_dir);

    int captured = 0;
    int displayed = 0;
    auto last_capture_time = std::chrono::steady_clock::now();
    const std::vector<int> jpeg_params = {cv::IMWRITE_JPEG_QUALITY, args.jpeg_quality};

    std::cout << (use_video ? "Video" : "Camera") << " opened. Starting capture...\n" << std::endl;

    while (g_running) {
        cv::Mat raw_frame;

        if (use_video) {
            if (!video_cap.read(raw_frame) || raw_frame.empty()) {
                std::cout << "\nEnd of video reached." << std::endl;
                break;  // end of file
            }
        } else {
            auto frame = camera->grab_frame();
            if (!frame.is_valid()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            raw_frame = frame.image;
        }

        ++displayed;
        bool should_capture = false;

        // Video-file mode with no interval: capture every frame
        if (use_video && args.interval_ms == 0) {
            should_capture = true;
        }

        // Auto-interval mode
        if (args.interval_ms > 0) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - last_capture_time).count();
            if (elapsed >= args.interval_ms) {
                should_capture = true;
                last_capture_time = now;
            }
        }

        // Show preview
        if (args.show_preview) {
            cv::Mat preview;
            // Resize for display if too large
            if (raw_frame.cols > 1280) {
                double scale = 1280.0 / raw_frame.cols;
                cv::resize(raw_frame, preview, cv::Size(), scale, scale);
            } else {
                preview = raw_frame;
            }

            // Add HUD overlay
            std::string hud = "Captured: " + std::to_string(captured);
            if (args.count > 0)
                hud += " / " + std::to_string(args.count);
            hud += "  |  Frame: " + std::to_string(displayed);

            cv::putText(preview, hud, cv::Point(10, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

            if (args.interval_ms == 0) {
                cv::putText(preview, "SPACE=capture  Q=quit", cv::Point(10, 60),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 200, 200), 1);
            }

            cv::imshow("EdgeAI Capture", preview);

            int key = cv::waitKey(1);
            if (key == 'q' || key == 'Q' || key == 27) {  // Q or ESC
                break;
            }
            if (key == ' ') {  // SPACE in interactive mode
                should_capture = true;
            }
        } else {
            // Headless mode — small sleep to avoid busy-loop
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        // Save frame
        if (should_capture) {
            std::string path = make_filename(args.output_dir, captured);
            if (cv::imwrite(path, raw_frame, jpeg_params)) {
                ++captured;
                std::cout << "  [" << captured << "] Saved: " << path
                          << "  (" << raw_frame.cols << "x" << raw_frame.rows << ")"
                          << std::endl;
            } else {
                std::cerr << "  ERROR: Failed to save " << path << std::endl;
            }

            // Check count limit
            if (args.count > 0 && captured >= args.count) {
                std::cout << "\nTarget count reached (" << args.count << ")." << std::endl;
                break;
            }
        }
    }

    if (use_video) {
        video_cap.release();
    } else {
        camera->close();
    }
    if (args.show_preview) {
        cv::destroyAllWindows();
    }

    std::cout << "\n════════════════════════════════════════════" << std::endl;
    std::cout << "  Capture complete: " << captured << " images saved to "
              << args.output_dir << "/" << std::endl;
    std::cout << "  Total frames displayed: " << displayed << std::endl;
    std::cout << "════════════════════════════════════════════\n" << std::endl;

    return 0;
}
