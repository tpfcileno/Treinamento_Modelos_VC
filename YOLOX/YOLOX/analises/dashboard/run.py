#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Serve a lightweight, dependency-free dashboard for YOLOX metric CSVs."""

import argparse
import csv
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "YOLOX_outputs" / "yolox_m_integra" / "metrics"
SUMMARY_FILE = "summary_metrics_by_confidence.csv"
CLASS_FILE = "per_class_metrics_by_confidence.csv"
CONFUSION_PREFIX = "confusion_matrix_conf_"
REQUIRED_FILES = (SUMMARY_FILE, CLASS_FILE)


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOX evaluation metrics dashboard")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing metrics CSV files.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Interface to bind.")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port.")
    return parser.parse_args()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def load_dashboard_data(input_dir):
    input_dir = input_dir.resolve()
    missing = [name for name in REQUIRED_FILES if not (input_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing required metrics CSV(s) in {input_dir}: {', '.join(missing)}"
        )

    summary = read_csv(input_dir / SUMMARY_FILE)
    class_metrics = read_csv(input_dir / CLASS_FILE)
    confusion = {}
    for path in sorted(input_dir.glob(f"{CONFUSION_PREFIX}*.csv")):
        threshold = path.stem.removeprefix(CONFUSION_PREFIX)
        rows = read_csv(path)
        if not rows or "actual\\predicted" not in rows[0]:
            raise ValueError(f"Unexpected confusion matrix CSV format: {path.name}")
        labels = [column for column in rows[0] if column != "actual\\predicted"]
        row_labels = [row["actual\\predicted"] for row in rows]
        if labels != row_labels:
            raise ValueError(f"Confusion matrix is not square or labels differ: {path.name}")
        confusion[threshold] = {
            "labels": labels,
            "matrix": [[int(row[label]) for label in labels] for row in rows],
        }

    if not summary or not class_metrics:
        raise ValueError("Summary and per-class metric CSVs must contain rows.")
    if not confusion:
        raise FileNotFoundError(f"No confusion matrix CSVs found in {input_dir}")

    return {
        "summary": summary,
        "classes": class_metrics,
        "confusion": confusion,
        "source": str(input_dir),
    }


HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>YOLOX metrics dashboard</title>
<style>
:root{color-scheme:dark;--bg:#0b1220;--panel:#121d30;--line:#25354d;--text:#e5edf7;--muted:#9aabc0;--accent:#59b7ff;--green:#42d392}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1500px;margin:auto;padding:28px}h1{margin:0;font-size:28px}h2{font-size:17px;margin:0 0 16px}p{color:var(--muted)}
.top{display:flex;justify-content:space-between;align-items:end;gap:18px;margin-bottom:20px}.top p{margin:5px 0 0}
.controls{display:flex;align-items:center;gap:10px;color:var(--muted)}select{background:#0b1424;color:var(--text);border:1px solid var(--line);border-radius:8px;padding:9px 12px}
.cards{display:grid;grid-template-columns:repeat(4,minmax(130px,1fr));gap:12px;margin:18px 0}
.card,.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px}.card{padding:16px}.card label{display:block;color:var(--muted);font-size:12px}.card strong{display:block;font-size:25px;margin-top:3px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.panel{padding:18px;min-width:0}.wide{grid-column:1/-1}
canvas{display:block;width:100%;height:300px}#classChart{height:920px}
.chart-legend{display:flex;flex-wrap:wrap;gap:8px 16px;margin:4px 0 10px;color:#c6d5e7;font-size:12px}
.chart-legend span{display:inline-flex;align-items:center;gap:6px}.chart-legend i{width:11px;height:11px;border-radius:2px}
.scroll{overflow:auto;max-height:520px}.matrix-scroll{overflow:auto}.matrix-scroll canvas{width:auto;height:auto;max-width:none}
table{width:100%;border-collapse:collapse;white-space:nowrap}th,td{text-align:right;padding:9px 10px;border-bottom:1px solid var(--line)}th{position:sticky;top:0;background:#17263d;color:#c8d8eb;font-size:12px}th:first-child,td:first-child{text-align:left}
.legend{font-size:12px;color:var(--muted);margin-top:8px}.status{color:var(--muted);font-size:12px}.error{color:#ff8585}.tooltip{position:fixed;display:none;background:#07101e;border:1px solid #435671;padding:7px 10px;border-radius:6px;pointer-events:none;z-index:5}
@media(max-width:850px){main{padding:16px}.top{align-items:flex-start;flex-direction:column}.cards{grid-template-columns:repeat(2,minmax(120px,1fr))}.grid{grid-template-columns:1fr}.wide{grid-column:auto}}
</style>
</head>
<body><main>
<header class="top"><div><h1>YOLOX evaluation dashboard</h1><p id="source">Loading report data...</p></div>
<label class="controls">Confidence threshold <select id="threshold"></select></label></header>
<section class="cards" id="cards"></section>
<section class="grid">
<article class="panel"><h2>Precision / recall / F1 by confidence</h2><canvas id="thresholdChart"></canvas><div class="legend">Metrics use micro-averaging across classes.</div></article>
<article class="panel"><h2>Per-class comparison at selected threshold</h2><div id="classLegend" class="chart-legend"></div><canvas id="classChart"></canvas><div class="legend">AP/AR are COCO metrics and do not change with this confidence threshold; F1 uses the selected confidence and IoU. Classes without test annotations are marked "No GT".</div></article>
<article class="panel wide"><h2>Confusion matrix - actual rows, predicted columns</h2><div class="matrix-scroll"><canvas id="matrix"></canvas></div><div class="legend">Cell color shows share of the actual class row; background row = false positives, background column = missed objects. Hover cells for details.</div></article>
<article class="panel wide"><h2>Per-class detail</h2><div class="scroll"><table id="classTable"></table></div></article>
<article class="panel wide"><h2>Summary by threshold</h2><div class="scroll"><table id="summaryTable"></table></div></article>
</section>
<p class="status" id="status"></p><div class="tooltip" id="tooltip"></div>
</main>
<script>
const $=s=>document.querySelector(s), number=(v,d=3)=>v===""||v==null?"—":Number(v).toFixed(d), percent=v=>v===""||v==null?"—":(Number(v)*100).toFixed(1)+"%";
let data, activeMatrixLayout=null;
function table(el,headers,rows,formats={}){el.innerHTML="<thead><tr>"+headers.map(h=>`<th>${h}</th>`).join("")+"</tr></thead><tbody>"+rows.map(r=>"<tr>"+headers.map(h=>`<td>${formats[h]?formats[h](r[h]):(r[h]??"—")}</td>`).join("")+"</tr>").join("")+"</tbody>"}
function selectedThreshold(){return $("#threshold").value}
function classRows(){return data.classes.filter(r=>Number(r.confidence_threshold)===Number(selectedThreshold()))}
function makeCards(summary){const defs=[["mAP50:95",percent(summary.mAP50_95_COCO)],["AP50",percent(summary.AP50_COCO)],["Micro F1",percent(summary.micro_f1)],["Micro precision / recall",percent(summary.micro_precision)+" / "+percent(summary.micro_recall)]];$("#cards").innerHTML=defs.map(([k,v])=>`<article class="card"><label>${k}</label><strong>${v}</strong></article>`).join("")}
function drawLineChart(canvas,rows){const ctx=canvas.getContext("2d"),dpr=devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;canvas.width=w*dpr;canvas.height=h*dpr;ctx.scale(dpr,dpr);ctx.clearRect(0,0,w,h);const pad={l:48,r:16,t:18,b:46},cw=w-pad.l-pad.r,ch=h-pad.t-pad.b,series=[["micro_precision","#57b7ff","Precision"],["micro_recall","#42d392","Recall"],["micro_f1","#ffbe55","F1"]];ctx.font="11px system-ui";ctx.strokeStyle="#2a3a50";ctx.fillStyle="#a5b5c9";for(let i=0;i<=4;i++){let y=pad.t+ch*i/4;ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(w-pad.r,y);ctx.stroke();ctx.fillText((1-i/4).toFixed(2),4,y+4)}rows.forEach((r,i)=>{let x=pad.l+(rows.length===1?cw/2:cw*i/(rows.length-1));ctx.fillText((Number(r.confidence_threshold)*100)+"%",x-12,h-17)});series.forEach(([key,color,label],si)=>{ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=2;ctx.beginPath();rows.forEach((r,i)=>{let x=pad.l+(rows.length===1?cw/2:cw*i/(rows.length-1)),y=pad.t+(1-Number(r[key]))*ch;i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke();rows.forEach((r,i)=>{let x=pad.l+(rows.length===1?cw/2:cw*i/(rows.length-1)),y=pad.t+(1-Number(r[key]))*ch;ctx.beginPath();ctx.arc(x,y,3,0,Math.PI*2);ctx.fill()});ctx.fillText(label,pad.l+si*90,pad.t+si*16)});}
function drawClassChart(canvas,rows){
const ctx=canvas.getContext("2d"),dpr=devicePixelRatio||1,w=canvas.clientWidth;
const metrics=[["AP50_95","#3486eb","AP50:95"],["AP50","#54c9a8","AP50"],["AR50_95","#f4b942","AR50:95"],["f1","#ed7d9a","F1"]];
ctx.font="11px system-ui";const longestLabel=Math.max(0,...rows.map(r=>ctx.measureText(r.class).width));
$("#classLegend").innerHTML=metrics.map(([,color,label])=>`<span><i style="background:${color}"></i>${label}</span>`).join("");
const left=Math.min(210,Math.max(142,longestLabel+18)),right=52,top=68,bottom=42,rowH=62;
const h=Math.max(480,top+rows.length*rowH+bottom);canvas.style.height=h+"px";canvas.width=w*dpr;canvas.height=h*dpr;
ctx.scale(dpr,dpr);ctx.clearRect(0,0,w,h);
const plotWidth=w-left-right,plotBottom=top+rows.length*rowH;
ctx.font="11px system-ui";ctx.textBaseline="middle";
for(let tick=0;tick<=4;tick++){const value=tick/4,x=left+plotWidth*value;ctx.beginPath();ctx.strokeStyle="#2a3a50";ctx.lineWidth=1;ctx.moveTo(x,top-5);ctx.lineTo(x,plotBottom);ctx.stroke();ctx.fillStyle="#9aabc0";ctx.textAlign="center";ctx.fillText(Math.round(value*100)+"%",x,top-19)}
rows.forEach((row,index)=>{
const rowY=top+index*rowH,rowCenter=rowY+rowH/2,hasGroundTruth=Number(row.ground_truth)>0;
ctx.textAlign="right";ctx.fillStyle="#c6d5e7";ctx.font="11px system-ui";ctx.fillText(row.class,left-12,rowCenter);
if(!hasGroundTruth){ctx.textAlign="left";ctx.fillStyle="#9aabc0";ctx.fillText("No GT",left+6,rowCenter);return}
metrics.forEach(([key,color],metricIndex)=>{
const raw=row[key],value=Number(raw);if(raw===""||raw==null||!Number.isFinite(value))return;
const clamped=Math.max(0,Math.min(1,value)),barHeight=8,barY=rowY+3+metricIndex*14;
const barWidth=plotWidth*clamped;ctx.fillStyle=color;ctx.fillRect(left,barY,barWidth,barHeight);
const label=(clamped*100).toFixed(0)+"%",labelWidth=ctx.measureText(label).width;
ctx.font="10px system-ui";
if(barWidth>labelWidth+8){ctx.textAlign="right";ctx.fillStyle="#08111f";ctx.fillText(label,left+barWidth-4,barY+barHeight/2)}
else{ctx.textAlign="left";ctx.fillStyle="#c6d5e7";ctx.fillText(label,left+barWidth+4,barY+barHeight/2)}
});
});
ctx.textAlign="left";ctx.textBaseline="alphabetic";
}
function drawMatrix(canvas,matrix){const ctx=canvas.getContext("2d"),labels=matrix.labels,values=matrix.matrix,n=labels.length,cell=62,left=175,top=205,right=94,bottom=52,w=left+n*cell+right,h=top+n*cell+bottom,dpr=devicePixelRatio||1;canvas.width=w*dpr;canvas.height=h*dpr;canvas.style.width=w+"px";canvas.style.height=h+"px";ctx.scale(dpr,dpr);ctx.clearRect(0,0,w,h);let rowTotals=values.map(row=>row.reduce((a,b)=>a+b,0)),maxShare=1;ctx.font="12px system-ui";ctx.textBaseline="middle";
labels.forEach((label,r)=>{let y=top+r*cell;ctx.fillStyle="#c6d5e7";ctx.textAlign="right";ctx.fillText(label,left-12,y+cell/2);labels.forEach((pred,c)=>{let share=rowTotals[r]?values[r][c]/rowTotals[r]:0,x=left+c*cell;ctx.fillStyle=blue(share);ctx.fillRect(x,y,cell-1,cell-1);ctx.fillStyle=share>.58?"white":"#0b1c2d";ctx.textAlign="center";ctx.font="12px system-ui";ctx.fillText(String(values[r][c]),x+cell/2,y+cell/2-7);ctx.font="10px system-ui";ctx.fillText(rowTotals[r]?percent(share):"—",x+cell/2,y+cell/2+9)});});
labels.forEach((label,c)=>{let x=left+c*cell+cell/2;ctx.save();ctx.translate(x,top-12);ctx.rotate(-Math.PI/4);ctx.textAlign="left";ctx.fillStyle="#c6d5e7";ctx.font="12px system-ui";ctx.fillText(label,0,0);ctx.restore()});
ctx.save();ctx.translate(24,top+n*cell/2);ctx.rotate(-Math.PI/2);ctx.textAlign="center";ctx.fillStyle="#dce8f5";ctx.fillText("Actual class",0,0);ctx.restore();ctx.textAlign="center";ctx.fillText("Predicted class",left+n*cell/2,h-18);
let bx=left+n*cell+27,by=top,bh=n*cell;for(let i=0;i<bh;i++){ctx.fillStyle=blue(1-i/bh);ctx.fillRect(bx,by+i,22,1)}ctx.textAlign="left";ctx.fillStyle="#c6d5e7";for(let i=0;i<=4;i++){let p=1-i/4;ctx.fillText(Math.round(p*100)+"%",bx+29,by+(1-p)*bh)}ctx.fillText("Row share",bx-8,top-16);
activeMatrixLayout={left,top,cell,labels,values,rowTotals,canvas};}
function blue(v){let stops=[[247,251,255],[222,235,247],[198,219,239],[158,202,225],[107,174,214],[66,146,198],[33,113,181],[8,81,156],[8,48,107]],p=Math.max(0,Math.min(1,v))*(stops.length-1),i=Math.floor(p),j=Math.min(i+1,stops.length-1),t=p-i;return "rgb("+stops[i].map((x,k)=>Math.round(x*(1-t)+stops[j][k]*t)).join(",")+")"}
function render(){const threshold=selectedThreshold(),summary=data.summary.find(r=>Number(r.confidence_threshold)===Number(threshold)),classes=classRows();makeCards(summary);drawLineChart($("#thresholdChart"),data.summary);drawClassChart($("#classChart"),classes);let matrix=data.confusion[String(Number(threshold)*100)]||data.confusion[String(Number(threshold)*100).replace(".0","")];if(matrix)drawMatrix($("#matrix"),matrix);let classHeaders=["class","ground_truth","predictions","tp","fp","fn","precision","recall","f1","AP50_95","AP50","AP75","AR50_95"];table($("#classTable"),classHeaders,classes,{precision:percent,recall:percent,f1:percent,AP50_95:percent,AP50:percent,AP75:percent,AR50_95:percent});let summaryHeaders=["confidence_threshold","tp","fp","fn","micro_precision","micro_recall","micro_f1","macro_f1"];table($("#summaryTable"),summaryHeaders,data.summary,{confidence_threshold:percent,micro_precision:percent,micro_recall:percent,micro_f1:percent,macro_f1:percent});}
async function init(){try{const response=await fetch("/api/data");if(!response.ok)throw new Error(await response.text());data=await response.json();$("#source").textContent="Source CSVs: "+data.source;const thresholds=data.summary.map(r=>r.confidence_threshold);$("#threshold").innerHTML=thresholds.map(v=>`<option value="${Number(v)}">${percent(v)}</option>`).join("");$("#threshold").value=thresholds.includes("0.5")?"0.5":String(thresholds[0]);$("#threshold").addEventListener("change",render);window.addEventListener("resize",render);render();$("#status").textContent=`Loaded ${data.classes.length} class/threshold rows and ${Object.keys(data.confusion).length} confusion matrices.`}catch(error){$("#status").className="status error";$("#status").textContent=error.message}}
$("#matrix").addEventListener("mousemove",event=>{if(!activeMatrixLayout)return;let rect=event.currentTarget.getBoundingClientRect(),x=event.clientX-rect.left,y=event.clientY-rect.top,l=activeMatrixLayout,c=Math.floor((x-l.left)/l.cell),r=Math.floor((y-l.top)/l.cell);if(r<0||c<0||r>=l.labels.length||c>=l.labels.length){$("#tooltip").style.display="none";return}let count=l.values[r][c],share=l.rowTotals[r]?count/l.rowTotals[r]:0,tip=$("#tooltip");tip.textContent=`Actual: ${l.labels[r]} | Predicted: ${l.labels[c]} | Count: ${count} | Row share: ${percent(share)}`;tip.style.display="block";tip.style.left=(event.clientX+12)+"px";tip.style.top=(event.clientY+12)+"px"});$("#matrix").addEventListener("mouseleave",()=>$("#tooltip").style.display="none");
init();
</script></body></html>"""


def make_handler(input_dir):
    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            route = urlparse(self.path).path
            if route == "/":
                body = HTML_PAGE.encode("utf-8")
                content_type = "text/html; charset=utf-8"
            elif route == "/api/data":
                try:
                    body = json.dumps(load_dashboard_data(input_dir)).encode("utf-8")
                except (OSError, ValueError, csv.Error) as error:
                    body = json.dumps({"error": str(error)}).encode("utf-8")
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                content_type = "application/json; charset=utf-8"
            else:
                body = b"Not found"
                content_type = "text/plain; charset=utf-8"
                self.send_response(404)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format_string, *args):
            print(f"{self.address_string()} - {format_string % args}")

    return DashboardHandler


def main():
    args = parse_args()
    input_dir = args.input_dir.resolve()
    load_dashboard_data(input_dir)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(input_dir))
    print(f"Dashboard available at http://{args.host}:{args.port}")
    print(f"Reading metrics from {input_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
