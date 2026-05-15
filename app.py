"""
Vehicle Detection Dashboard — Cloud Version
============================================
Versi ringan tanpa fitur deteksi.
Hanya menampilkan riwayat sesi dari Supabase.

Deploy ke Render:
  Build Command : pip install -r requirements.txt
  Start Command : gunicorn app:app

Environment variables di Render:
  SUPABASE_URL = https://xxxxx.supabase.co
  SUPABASE_KEY = your_anon_key_here
"""

import os
from flask import Flask, render_template, jsonify

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = Flask(__name__)


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


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/sessions")
def api_sessions():
    sb = get_supabase()
    if not sb:
        return jsonify({"error": "Supabase belum dikonfigurasi. Periksa environment variables."}), 500
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

    d           = res.data
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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)