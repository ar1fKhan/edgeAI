"""
EdgeAI Local Dashboard — Flask Application

Provides real-time monitoring of the defect detection pipeline.
All data served from local SQLite database — NO cloud.

Endpoints:
    GET  /                    → Dashboard UI
    GET  /api/stats           → Current pipeline statistics
    GET  /api/daily           → Daily stats (last 30 days)
    GET  /api/defects         → Recent defect records
    GET  /api/distribution    → Defect type distribution
    GET  /api/defect-image/<id> → Serve defect image

Usage:
    python app.py --db ../../data/defects.db --port 5000
"""

import argparse
import json
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from flask import Flask, jsonify, render_template_string, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

DB_PATH = "../../data/defects.db"
DEFECT_IMAGE_DIR = "../../data/defects"

# ── Dashboard HTML (embedded for zero-dependency deployment) ────

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EdgeAI — Paint Can Quality Inspector</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: #0f1419;
            color: #e1e8ed;
            min-height: 100vh;
        }
        .header {
            background: linear-gradient(135deg, #1a1f2e 0%, #2d1b69 100%);
            padding: 20px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid #7c3aed;
        }
        .header h1 { font-size: 1.5em; color: #a78bfa; }
        .header .status {
            display: flex; align-items: center; gap: 8px;
            font-size: 0.9em; color: #4ade80;
        }
        .header .status .dot {
            width: 10px; height: 10px; border-radius: 50%;
            background: #4ade80; animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }

        /* KPI Cards */
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px; margin-bottom: 24px;
        }
        .kpi-card {
            background: #1a1f2e;
            border-radius: 12px;
            padding: 20px;
            border: 1px solid #2d3748;
            text-align: center;
        }
        .kpi-card .value {
            font-size: 2em; font-weight: 700;
            background: linear-gradient(to right, #a78bfa, #60a5fa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .kpi-card .label { font-size: 0.85em; color: #9ca3af; margin-top: 4px; }
        .kpi-card.danger .value { background: linear-gradient(to right, #f87171, #fb923c); -webkit-background-clip: text; }
        .kpi-card.success .value { background: linear-gradient(to right, #4ade80, #22d3ee); -webkit-background-clip: text; }

        /* Charts */
        .charts-grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 16px; margin-bottom: 24px;
        }
        .chart-card {
            background: #1a1f2e;
            border-radius: 12px;
            padding: 20px;
            border: 1px solid #2d3748;
        }
        .chart-card h3 { color: #a78bfa; margin-bottom: 16px; font-size: 1em; }

        /* Defect Table */
        .table-card {
            background: #1a1f2e;
            border-radius: 12px;
            padding: 20px;
            border: 1px solid #2d3748;
        }
        .table-card h3 { color: #a78bfa; margin-bottom: 16px; }
        table { width: 100%; border-collapse: collapse; }
        th { text-align: left; padding: 10px; color: #9ca3af; border-bottom: 1px solid #2d3748; font-size: 0.85em; }
        td { padding: 10px; border-bottom: 1px solid #1f2937; font-size: 0.9em; }
        .badge {
            display: inline-block; padding: 3px 10px; border-radius: 12px;
            font-size: 0.8em; font-weight: 600;
        }
        .badge-reject { background: #7f1d1d; color: #fca5a5; }
        .badge-pass { background: #14532d; color: #86efac; }
        .badge-review { background: #713f12; color: #fde047; }
        .badge-dent { background: #1e3a5f; color: #93c5fd; }
        .badge-wrong_label { background: #5b21b6; color: #c4b5fd; }
        .badge-missing_label { background: #9a3412; color: #fdba74; }
        .badge-seal_defect { background: #065f46; color: #6ee7b7; }
        .badge-color_mismatch { background: #831843; color: #f9a8d4; }

        .footer {
            text-align: center; padding: 20px;
            color: #4b5563; font-size: 0.8em;
        }
        @media (max-width: 768px) {
            .charts-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🏭 EdgeAI — Paint Can Quality Inspector</h1>
        <div class="status">
            <div class="dot"></div>
            <span>LOCAL PROCESSING — NO CLOUD</span>
        </div>
    </div>

    <div class="container">
        <!-- KPI Cards -->
        <div class="kpi-grid" id="kpiGrid">
            <div class="kpi-card"><div class="value" id="totalInspected">—</div><div class="label">Total Inspected</div></div>
            <div class="kpi-card success"><div class="value" id="passRate">—</div><div class="label">Pass Rate</div></div>
            <div class="kpi-card danger"><div class="value" id="defectRate">—</div><div class="label">Defect Rate</div></div>
            <div class="kpi-card"><div class="value" id="avgLatency">—</div><div class="label">Avg Latency (ms)</div></div>
            <div class="kpi-card"><div class="value" id="throughput">—</div><div class="label">Throughput (FPS)</div></div>
            <div class="kpi-card danger"><div class="value" id="totalRejects">—</div><div class="label">Total Rejects</div></div>
        </div>

        <!-- Charts -->
        <div class="charts-grid">
            <div class="chart-card">
                <h3>📈 Daily Inspection Trend</h3>
                <canvas id="trendChart"></canvas>
            </div>
            <div class="chart-card">
                <h3>🎯 Defect Distribution</h3>
                <canvas id="distributionChart"></canvas>
            </div>
        </div>

        <!-- Recent Defects Table -->
        <div class="table-card">
            <h3>🔍 Recent Defects</h3>
            <table>
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Frame</th>
                        <th>Defect Type</th>
                        <th>Confidence</th>
                        <th>Verdict</th>
                        <th>Latency</th>
                    </tr>
                </thead>
                <tbody id="defectTable"></tbody>
            </table>
        </div>
    </div>

    <div class="footer">
        EdgeAI Defect Detection System v1.0 — All data processed locally on edge device
    </div>

    <script>
        let trendChart, distributionChart;

        async function fetchJSON(url) {
            const res = await fetch(url);
            return res.json();
        }

        async function updateKPIs() {
            const stats = await fetchJSON('/api/stats');
            document.getElementById('totalInspected').textContent = stats.total_inspected?.toLocaleString() || '0';
            document.getElementById('passRate').textContent = stats.pass_rate ? stats.pass_rate.toFixed(1) + '%' : '—';
            document.getElementById('defectRate').textContent = stats.defect_rate ? stats.defect_rate.toFixed(1) + '%' : '—';
            document.getElementById('avgLatency').textContent = stats.avg_inference_ms ? stats.avg_inference_ms.toFixed(1) : '—';
            document.getElementById('throughput').textContent = stats.throughput_fps ? stats.throughput_fps.toFixed(0) : '—';
            document.getElementById('totalRejects').textContent = stats.total_rejects?.toLocaleString() || '0';
        }

        async function updateTrendChart() {
            const daily = await fetchJSON('/api/daily');
            const labels = daily.map(d => d.date).reverse();
            const passed = daily.map(d => d.total_passed).reverse();
            const defects = daily.map(d => d.total_defects).reverse();

            if (trendChart) trendChart.destroy();
            trendChart = new Chart(document.getElementById('trendChart'), {
                type: 'bar',
                data: {
                    labels,
                    datasets: [
                        { label: 'Passed', data: passed, backgroundColor: '#4ade8080', borderColor: '#4ade80', borderWidth: 1 },
                        { label: 'Defects', data: defects, backgroundColor: '#f8717180', borderColor: '#f87171', borderWidth: 1 }
                    ]
                },
                options: {
                    responsive: true,
                    scales: {
                        x: { stacked: true, ticks: { color: '#9ca3af' }, grid: { color: '#1f2937' } },
                        y: { stacked: true, ticks: { color: '#9ca3af' }, grid: { color: '#1f2937' } }
                    },
                    plugins: { legend: { labels: { color: '#e1e8ed' } } }
                }
            });
        }

        async function updateDistributionChart() {
            const dist = await fetchJSON('/api/distribution');
            const labels = dist.map(d => d.type);
            const values = dist.map(d => d.count);
            const colors = ['#60a5fa', '#c084fc', '#fb923c', '#4ade80', '#f472b6'];

            if (distributionChart) distributionChart.destroy();
            distributionChart = new Chart(document.getElementById('distributionChart'), {
                type: 'doughnut',
                data: {
                    labels,
                    datasets: [{
                        data: values,
                        backgroundColor: colors.slice(0, labels.length),
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    plugins: { legend: { position: 'bottom', labels: { color: '#e1e8ed', padding: 12 } } }
                }
            });
        }

        async function updateDefectTable() {
            const defects = await fetchJSON('/api/defects?limit=20');
            const tbody = document.getElementById('defectTable');
            tbody.innerHTML = defects.map(d => `
                <tr>
                    <td>${d.timestamp}</td>
                    <td>#${d.frame_id}</td>
                    <td><span class="badge badge-${d.defect_type}">${d.defect_type}</span></td>
                    <td>${(d.confidence * 100).toFixed(1)}%</td>
                    <td><span class="badge badge-${d.verdict.toLowerCase()}">${d.verdict}</span></td>
                    <td>${d.inference_ms.toFixed(1)} ms</td>
                </tr>
            `).join('');
        }

        async function refresh() {
            try {
                await Promise.all([
                    updateKPIs(),
                    updateTrendChart(),
                    updateDistributionChart(),
                    updateDefectTable()
                ]);
            } catch (e) {
                console.error('Refresh error:', e);
            }
        }

        // Initial load + auto-refresh every 5 seconds
        refresh();
        setInterval(refresh, 5000);
    </script>
</body>
</html>
"""


def get_db():
    """Get SQLite connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@app.route("/")
def dashboard():
    """Serve the dashboard UI."""
    return render_template_string(DASHBOARD_HTML)


@app.route("/api/stats")
def api_stats():
    """Get current pipeline statistics."""
    try:
        conn = get_db()
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) FROM inspections")
        total = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM inspections WHERE verdict = 'REJECT'")
        rejects = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM inspections WHERE verdict = 'PASS'")
        passed = cur.fetchone()[0]

        cur.execute("SELECT AVG(inference_ms) FROM inspections")
        avg_inf = cur.fetchone()[0] or 0

        cur.execute("SELECT AVG(total_ms) FROM inspections")
        avg_total = cur.fetchone()[0] or 0

        conn.close()

        defect_rate = (rejects / total * 100) if total > 0 else 0
        pass_rate = (passed / total * 100) if total > 0 else 0
        fps = (1000.0 / avg_total) if avg_total > 0 else 0

        return jsonify({
            "total_inspected": total,
            "total_rejects": rejects,
            "total_passed": passed,
            "defect_rate": round(defect_rate, 2),
            "pass_rate": round(pass_rate, 2),
            "avg_inference_ms": round(avg_inf, 2),
            "avg_total_ms": round(avg_total, 2),
            "throughput_fps": round(fps, 1)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/daily")
def api_daily():
    """Get daily statistics for the last 30 days."""
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            SELECT 
                DATE(timestamp) as date,
                COUNT(*) as total,
                SUM(CASE WHEN verdict = 'REJECT' THEN 1 ELSE 0 END) as defects,
                SUM(CASE WHEN verdict = 'PASS' THEN 1 ELSE 0 END) as passed,
                AVG(inference_ms) as avg_inference
            FROM inspections
            WHERE timestamp >= DATE('now', '-30 days')
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
        """)
        rows = cur.fetchall()
        conn.close()

        return jsonify([{
            "date": row["date"],
            "total_inspected": row["total"],
            "total_defects": row["defects"],
            "total_passed": row["passed"],
            "defect_rate": round((row["defects"] / row["total"] * 100) if row["total"] > 0 else 0, 2),
            "avg_inference_ms": round(row["avg_inference"] or 0, 2)
        } for row in rows])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/defects")
def api_defects():
    """Get recent defect records."""
    from flask import request
    limit = request.args.get("limit", 50, type=int)

    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            SELECT i.id, i.frame_id, i.timestamp, d.defect_type, d.confidence,
                   i.verdict, i.inference_ms, i.image_path
            FROM inspections i
            JOIN detections d ON d.inspection_id = i.id
            WHERE i.verdict = 'REJECT'
            ORDER BY i.timestamp DESC
            LIMIT ?
        """, (limit,))
        rows = cur.fetchall()
        conn.close()

        return jsonify([{
            "id": row["id"],
            "frame_id": row["frame_id"],
            "timestamp": row["timestamp"],
            "defect_type": row["defect_type"],
            "confidence": row["confidence"],
            "verdict": row["verdict"],
            "inference_ms": row["inference_ms"],
            "image_path": row["image_path"]
        } for row in rows])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/distribution")
def api_distribution():
    """Get defect type distribution."""
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            SELECT defect_type, COUNT(*) as cnt
            FROM detections
            GROUP BY defect_type
            ORDER BY cnt DESC
        """)
        rows = cur.fetchall()
        conn.close()

        return jsonify([{
            "type": row["defect_type"],
            "count": row["cnt"]
        } for row in rows])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/defect-image/<int:inspection_id>")
def api_defect_image(inspection_id):
    """Serve a defect image from local storage."""
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT image_path FROM inspections WHERE id = ?", (inspection_id,))
        row = cur.fetchone()
        conn.close()

        if row and row["image_path"] and os.path.exists(row["image_path"]):
            return send_file(row["image_path"], mimetype="image/jpeg")
        return jsonify({"error": "Image not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def main():
    parser = argparse.ArgumentParser(description="EdgeAI Dashboard")
    parser.add_argument("--db", type=str, default="../../data/defects.db",
                        help="SQLite database path")
    parser.add_argument("--images", type=str, default="../../data/defects",
                        help="Defect images directory")
    parser.add_argument("--port", type=int, default=5000,
                        help="Dashboard port")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Dashboard host")
    args = parser.parse_args()

    global DB_PATH, DEFECT_IMAGE_DIR
    DB_PATH = args.db
    DEFECT_IMAGE_DIR = args.images

    print("=" * 50)
    print("  EdgeAI Dashboard")
    print("=" * 50)
    print(f"  Database:  {DB_PATH}")
    print(f"  Images:    {DEFECT_IMAGE_DIR}")
    print(f"  URL:       http://{args.host}:{args.port}")
    print(f"  Mode:      LOCAL (no cloud)")
    print("=" * 50)

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
