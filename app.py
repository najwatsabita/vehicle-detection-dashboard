"""
Vehicle Detection Dashboard — Flask Backend
============================================
Jalankan: python app.py --model ../yolov12.pt
Buka browser: http://localhost:8080

Konfigurasi .env (letakkan di folder NEW/ atau dashboard/):
    SUPABASE_URL=https://xxxxx.supabase.co
    SUPABASE_KEY=your_anon_key_here
"""

import os
import sys
import time
import threading
import argparse
from pathlib import Path
from datetime import datetime

import cv2

from flask import Flask, render_template, jsonify, Response, request

# Tambahkan folder parent (NEW/) ke sys.path agar shared_state
# dan vehicle_detector bisa diimport dari subfolder dashboard/
_PARENT = Path(__file__).parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

import shared_state as state
import vehicle_detector as vd

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).parent / ".env")
    load_dotenv(dotenv_path=_PARENT / ".env")
except ImportError:
    pass

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

UPLOAD_FOLDER  = str(_PARENT / "uploads")
RESULTS_FOLDER = str(_PARENT / "results")
ALLOWED_EXT    = {".mp4", ".avi", ".mov", ".mkv"}

Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
Path(RESULTS_FOLDER).mkdir(exist_ok=True)

_models          = {}
_device          = "cpu"
_proc_thread     = None
_ext_stop        = threading.Event()
_last_video_path = None
_last_conf       = 0.35


# ─── Encoder thread ───────────────────────────────────────────────────────────
def _encoder_worker():
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 70]
    while True:
        try:
            frame = state._encode_queue.get(timeout=1.0)
        except Exception:
            continue
        ret, buf = cv2.imencode(".jpg", frame, encode_params)
        if ret:
            with state._jpeg_lock:
                state._latest_jpeg = buf.tobytes()
        state._encode_queue.task_done()


threading.Thread(target=_encoder_worker, daemon=True, name="EncoderThread").start()


# ─── MJPEG generator ──────────────────────────────────────────────────────────
def _mjpeg_generator():
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    while True:
        with state._jpeg_lock:
            jpeg = state._latest_jpeg
        if jpeg is None:
            time.sleep(0.05)
            continue
        yield boundary + jpeg + b"\r\n"
        time.sleep(1.0 / 25)


# ─── Background detection thread ──────────────────────────────────────────────
def _run_detection(video_path: str, output_json: str, conf: float, model, keep_file: bool = False):
    global _proc_thread, _last_video_path, _last_conf
    try:
        state.update_status(state="processing", filename=Path(video_path).name, progress=0.0)

        def progress_callback(pct, counts):
            state.update_status(progress=round(pct, 1), counts=counts)

        result = vd.process_video(
            model             = model,
            device            = _device,
            video_path        = video_path,
            output_json       = output_json,
            conf_threshold    = conf,
            show_preview      = False,
            save_video        = False,
            progress_callback = progress_callback,
            external_stop     = _ext_stop,
        )
        _last_video_path = video_path
        _last_conf       = conf
        state.update_status(state="done", progress=100.0, result=result)
    except Exception as e:
        print(f"[ERROR] Detection thread: {e}")
        state.update_status(state="error", error=str(e))
        if not keep_file:
            try:
                if Path(video_path).exists():
                    os.remove(video_path)
            except Exception:
                pass
    finally:
        _proc_thread = None


# ─── Supabase Client ──────────────────────────────────────────────────────────
def get_supabase():
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_KEY", "").strip()
    if not url or not key:
        return None
    try:
        from supabase import create_client
        return create_client(url, key)
    except Exception as e:
        print(f"[ERROR] Supabase connect gagal: {e}")
        return None


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/info")
def api_info():
    return jsonify({
        "device":           _device.upper(),
        "available_models": list(_models.keys()),
        "model_loaded":     len(_models) > 0,
    })


@app.route("/upload", methods=["POST"])
def upload():
    global _proc_thread, _ext_stop

    if not _models:
        return jsonify({"error": "Model belum dimuat. Jalankan: python app.py --yolo ../yolov12.pt --rtdetr ../rt_detr.pt"}), 503

    if _proc_thread is not None and _proc_thread.is_alive():
        _ext_stop.set()
        _proc_thread.join(timeout=5)
        _ext_stop.clear()
        state.reset_status()

    if "video" not in request.files:
        return jsonify({"error": "Tidak ada file yang dikirim."}), 400

    f = request.files["video"]
    if not f.filename:
        return jsonify({"error": "Nama file kosong."}), 400

    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"Format tidak didukung: {ext}"}), 400

    model_key = request.form.get("model_key", "")
    if model_key not in _models:
        model_key = list(_models.keys())[0]
    selected_model = _models[model_key]

    conf       = float(request.form.get("conf", 0.35))
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name  = f"{ts}_{Path(f.filename).stem}{ext}"
    video_path = str(Path(UPLOAD_FOLDER) / safe_name)
    out_json   = str(Path(RESULTS_FOLDER) / f"{ts}_hasil.json")

    f.save(video_path)

    if _last_video_path and _last_video_path != video_path:
        try:
            if Path(_last_video_path).exists():
                os.remove(_last_video_path)
        except Exception:
            pass

    state.reset_status()
    _ext_stop.clear()

    _proc_thread = threading.Thread(
        target = _run_detection,
        args   = (video_path, out_json, conf, selected_model),
        daemon = True,
        name   = "DetectionThread",
    )
    _proc_thread.start()
    return jsonify({"ok": True, "model_used": model_key})


@app.route("/video_feed")
def video_feed():
    return Response(_mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    return jsonify(state.get_status())


@app.route("/stop", methods=["POST"])
def stop():
    _ext_stop.set()
    return jsonify({"ok": True})


@app.route("/redetect", methods=["POST"])
def redetect():
    global _proc_thread, _ext_stop

    if not _models:
        return jsonify({"error": "Model belum dimuat."}), 503

    if not _last_video_path or not Path(_last_video_path).exists():
        return jsonify({"error": "Video sebelumnya tidak tersedia. Silakan upload video baru."}), 400

    if _proc_thread is not None and _proc_thread.is_alive():
        _ext_stop.set()
        _proc_thread.join(timeout=5)
        _ext_stop.clear()
        state.reset_status()

    data      = request.get_json(silent=True) or {}
    model_key = data.get("model_key", "")
    if model_key not in _models:
        model_key = list(_models.keys())[0]
    selected_model = _models[model_key]

    conf     = float(data.get("conf", _last_conf))
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = str(Path(RESULTS_FOLDER) / f"{ts}_hasil.json")

    state.reset_status()
    _ext_stop.clear()

    _proc_thread = threading.Thread(
        target = _run_detection,
        args   = (_last_video_path, out_json, conf, selected_model, True),
        daemon = True,
        name   = "DetectionThread",
    )
    _proc_thread.start()
    return jsonify({"ok": True, "model_used": model_key})


@app.route("/api/sessions")
def api_sessions():
    """
    Ambil daftar semua sesi deteksi dari Supabase,
    diurutkan dari yang terbaru.
    """
    sb = get_supabase()
    if not sb:
        return jsonify({"error": "Supabase belum dikonfigurasi. Periksa file .env"}), 500

    try:
        res = (
            sb.table("detection_results")
            .select("id, video_name, created_at, total_unique, count_car, count_bus, count_truck, count_motorcycle")
            .order("created_at", desc=True)
            .execute()
        )
        return jsonify({"sessions": res.data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/session/<int:session_id>")
def api_session_detail(session_id):
    """
    Ambil detail lengkap satu sesi berdasarkan ID,
    format response sama persis dengan yang diharapkan dashboard.
    """
    sb = get_supabase()
    if not sb:
        return jsonify({"error": "Supabase belum dikonfigurasi"}), 500

    try:
        res = (
            sb.table("detection_results")
            .select("*")
            .eq("id", session_id)
            .single()
            .execute()
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    if not res.data:
        return jsonify({"error": f"Sesi ID {session_id} tidak ditemukan"}), 404

    d = res.data
    bus_count   = d.get("count_bus",   0)
    truck_count = d.get("count_truck", 0)

    return jsonify({
        "counts": {
            "car":        d.get("count_car",        0),
            "bus":        bus_count,
            "truck":      truck_count,
            "motorcycle": d.get("count_motorcycle", 0),
        },
        "total":               d.get("total_unique", 0),
        "large_vehicle_alert": (bus_count > 0 or truck_count > 0),
        "metadata": {
            "video_source": d.get("video_name",  "-"),
            "model_used":   d.get("model_used",  "-"),
            "resolution":   d.get("resolution",  "-"),
            "fps":          d.get("fps",          0),
            "duration":     d.get("duration",     0),
            "total_frames": d.get("total_frames", 0),
            "generated_at": d.get("created_at",  "-"),
        },
    })


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vehicle Detection Dashboard")
    parser.add_argument("--yolov12", default=None,
                        help="Path ke model YOLOv12 .pt (contoh: ../yolov12.pt)")
    parser.add_argument("--rtdetr", default=None,
                        help="Path ke model RT-DETR .pt (contoh: ../rt_detr.pt)")
    parser.add_argument("--device", "-d", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--port",   "-p", type=int, default=8080)
    args = parser.parse_args()

    _device = vd.resolve_device(args.device)

    def _resolve(p):
        path = Path(p)
        if not path.is_absolute():
            path = Path(__file__).parent / path
        return path

    if args.yolov12:
        p = _resolve(args.yolov12)
        print(f"[INFO] Memuat YOLOv12 dari {p}")
        _models["yolov12"] = vd.load_model(str(p), _device)

    if args.rtdetr:
        p = _resolve(args.rtdetr)
        print(f"[INFO] Memuat RT-DETR dari {p}")
        _models["rtdetr"] = vd.load_model(str(p), _device)

    if not _models:
        print("[WARN] Tidak ada model yang dimuat. Gunakan --yolov12 dan/atau --rtdetr.")

    HOST = "0.0.0.0"
    PORT = args.port
    print("=" * 55)
    print("  Vehicle Detection Dashboard")
    print(f"  Buka    : http://localhost:{PORT}")
    print(f"  Models  : {list(_models.keys()) or 'tidak ada'}")
    print(f"  Device  : {_device.upper()}")
    print("=" * 55)
    app.run(host=HOST, port=PORT, threaded=True, use_reloader=False, debug=False)
