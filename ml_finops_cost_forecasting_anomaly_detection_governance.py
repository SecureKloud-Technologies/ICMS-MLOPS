import math
import boto3
import statistics
import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import date, datetime, timedelta
from botocore.exceptions import ClientError
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# ==================== Configuration ====================
SENDER_EMAIL = "mbabinayaa@gmail.com"
RECEIVER_EMAIL = "mbabinayaa@gmail.com"
SES_REGION = "ap-south-1"
LOW_CPU_THRESHOLD = 10.0
CPU_LOOKBACK_DAYS = 14
ANOMALY_WINDOW_DAYS = 30
ANOMALY_CONTAMINATION = 0.1
ANOMALY_MIN_COST = 1.0
ANOMALY_MIN_HISTORY_DAYS = 10
ANOMALY_SPIKE_THRESHOLD_PCT = 20.0
OLD_SNAPSHOT_DAYS = 90
MOM_INCREASE_THRESHOLD_PCT = 30.0
TOP_N_SERVICES = 5
FORECAST_HISTORY_DAYS = 90
FORECAST_HORIZON_DAYS = 7
N_BACKTEST_ORIGINS = 90
ORIGIN_STRIDE_DAYS = 1
MIN_HISTORY_FOR_SELECTION = 40
EPS = 1e-6
SERVICE_NAMES = {"EC2": "Amazon Elastic Compute Cloud - Compute", "RDS": "Amazon Relational Database Service",
                 "EKS": "Amazon Elastic Container Service for Kubernetes",
                 "S3": "Amazon Simple Storage Service", "ELB": "Amazon Elastic Load Balancing"}
ce = boto3.client("ce")
ec2 = boto3.client("ec2")
cloudwatch = boto3.client("cloudwatch")
ses = boto3.client("ses", region_name=SES_REGION)

# ==================== Cost Analysis ====================
def month_bounds():
    today = date.today()
    this_start = today.replace(day=1)
    prev_start = (this_start - timedelta(days=1)).replace(day=1)
    return this_start, today + timedelta(days=1), prev_start, this_start
def get_elapsed_days(cur_start):
    return (date.today() - cur_start).days + 1
def ce_query(start, end, granularity="MONTHLY", group_by=None, filter_=None):
    kwargs = {"TimePeriod": {"Start": str(start), "End": str(end)},
              "Granularity": granularity, "Metrics": ["UnblendedCost"]}
    if group_by:
        kwargs["GroupBy"] = [{"Type": "DIMENSION", "Key": group_by}]
    if filter_:
        kwargs["Filter"] = filter_
    try:
        return ce.get_cost_and_usage(**kwargs)
    except ClientError as e:
        print(f"Cost Explorer query failed: {e}")
        return None
def get_total_cost(start, end):
    resp = ce_query(start, end)
    if not resp:
        return 0.0
    return float(resp["ResultsByTime"][0]["Total"]["UnblendedCost"]["Amount"])
def get_costs_by_service(start, end):
    resp = ce_query(start, end, group_by="SERVICE")
    costs = {}
    if resp:
        for g in resp["ResultsByTime"][0]["Groups"]:
            costs[g["Keys"][0]] = float(g["Metrics"]["UnblendedCost"]["Amount"])
    return costs
def get_nat_gateway_cost(start, end):
    resp = ce_query(start, end, group_by="USAGE_TYPE")
    total = 0.0
    if resp:
        for g in resp["ResultsByTime"][0]["Groups"]:
            if "NatGateway" in g["Keys"][0]:
                total += float(g["Metrics"]["UnblendedCost"]["Amount"])
    return total
def get_nat_gateway_daily_costs(start, end):
    resp = ce_query(start, end, granularity="DAILY", group_by="USAGE_TYPE")
    daily = []
    if resp:
        for day in resp["ResultsByTime"]:
            total = sum(float(g["Metrics"]["UnblendedCost"]["Amount"])
                        for g in day["Groups"] if "NatGateway" in g["Keys"][0])
            daily.append(total)
    return daily
def get_usage_type_cost(start, end, match_substr):
    resp = ce_query(start, end, group_by="USAGE_TYPE")
    total = 0.0
    if resp:
        for g in resp["ResultsByTime"][0]["Groups"]:
            if match_substr in g["Keys"][0]:
                total += float(g["Metrics"]["UnblendedCost"]["Amount"])
    return total
def project_to_full_month(mtd_cost, cur_start):
    today = date.today()
    elapsed_days = get_elapsed_days(cur_start)
    days_in_month = ((cur_start.replace(day=28) + timedelta(days=4)).replace(day=1) - cur_start).days
    if elapsed_days <= 0:
        return mtd_cost
    return mtd_cost / elapsed_days * days_in_month
def build_service_comparison(cur_start, cur_end, prev_start, prev_end):
    n_days = get_elapsed_days(cur_start)
    comparable_prev_end = min(prev_start + timedelta(days=n_days), prev_end)
    cur_costs = get_costs_by_service(cur_start, cur_end)
    prev_costs = get_costs_by_service(prev_start, comparable_prev_end)
    rows = [(label, cur_costs.get(name, 0.0), prev_costs.get(name, 0.0))
            for label, name in SERVICE_NAMES.items()]
    rows.append(("NAT Gateway", get_nat_gateway_cost(cur_start, cur_end),
                 get_nat_gateway_cost(prev_start, comparable_prev_end)))
    return rows, cur_costs, n_days
def get_mom_anomalies(service_rows, threshold=MOM_INCREASE_THRESHOLD_PCT):
    anomalies = []
    for label, cur, prev in service_rows:
        if prev <= 0:
            continue
        change_pct = (cur - prev) / prev * 100
        if change_pct >= threshold:
            anomalies.append({"label": label, "cur": cur, "prev": prev, "change_pct": change_pct})
    return sorted(anomalies, key=lambda a: a["change_pct"], reverse=True)
def top_services(cur_costs, n=TOP_N_SERVICES):
    return sorted(cur_costs.items(), key=lambda x: x[1], reverse=True)[:n]

# ==================== Cost Forecast ====================
def get_service_daily_history(days=FORECAST_HISTORY_DAYS):
    end = date.today() + timedelta(days=1)
    start = end - timedelta(days=days)
    dates = [start + timedelta(days=i) for i in range(days)]
    history = {label: [] for label in SERVICE_NAMES}
    resp = ce_query(start, end, granularity="DAILY", group_by="SERVICE")
    if resp:
        for day in resp["ResultsByTime"]:
            day_costs = {g["Keys"][0]: float(g["Metrics"]["UnblendedCost"]["Amount"]) for g in day["Groups"]}
            for label, name in SERVICE_NAMES.items():
                history[label].append(day_costs.get(name, 0.0))
    history["NAT Gateway"] = get_nat_gateway_daily_costs(start, end)
    return history, dates
def _window(values, end_idx, size):
    return values[max(0, end_idx - size + 1): end_idx + 1]
def _safe_mean(seq, default=0.0):
    return statistics.mean(seq) if seq else default
def f_naive7(values, dates, n):
    return max(0.0, sum(_window(values, n - 1, FORECAST_HORIZON_DAYS)))
def f_trimmed_median(values, dates, n):
    w = _window(values, n - 1, 14)
    return max(0.0, statistics.median(w) * FORECAST_HORIZON_DAYS) if w else 0.0
def f_weekday_seasonal(values, dates, n):
    if n < 21:
        return f_naive7(values, dates, n)
    buckets = {d: [] for d in range(7)}
    for i in range(max(0, n - 28), n):
        buckets[dates[i].weekday()].append(values[i])
    overall = _safe_mean(values[max(0, n - 28):n])
    total = 0.0
    for h in range(1, FORECAST_HORIZON_DAYS + 1):
        wd = (dates[n - 1] + timedelta(days=h)).weekday()
        total += _safe_mean(buckets[wd], overall)
    return max(0.0, total)
def f_damped_drift(values, dates, n):
    w = _window(values, n - 1, 28)
    if len(w) < 14:
        return f_naive7(values, dates, n)
    x = np.arange(len(w), dtype=float)
    y = np.array(w, dtype=float)
    try:
        slope, intercept = np.polyfit(x, y, 1)
    except Exception:
        return f_naive7(values, dates, n)
    phi, total, level = 0.85, 0.0, intercept + slope * (len(w) - 1)
    damp = 1.0
    for h in range(1, FORECAST_HORIZON_DAYS + 1):
        damp *= phi
        total += max(0.0, level + slope * damp * h)
    return max(0.0, total)
def f_longrun_mean(values, dates, n):
    return max(0.0, _safe_mean(_window(values, n - 1, 56)) * FORECAST_HORIZON_DAYS)
CANDIDATES = {
    "naive7": f_naive7,
    "trimmed_median": f_trimmed_median,
    "weekday_seasonal": f_weekday_seasonal,
    "damped_drift": f_damped_drift,
    "longrun_mean": f_longrun_mean,
}
def rolling_origins(n_days, n_origins=N_BACKTEST_ORIGINS, stride=ORIGIN_STRIDE_DAYS):
    origins = []
    for k in range(n_origins):
        c = n_days - FORECAST_HORIZON_DAYS - (k * stride)
        if c - 30 < FORECAST_HORIZON_DAYS:
            break
        origins.append(c)
    return origins
def backtest_service(values, dates, origins):
    if not origins:
        return {}
    errs = {name: [] for name in CANDIDATES}
    for c in origins:
        actual = sum(values[c: c + FORECAST_HORIZON_DAYS])
        for name, fn in CANDIDATES.items():
            try:
                pred = fn(values, dates, c)
            except Exception:
                pred = f_naive7(values, dates, c)
            errs[name].append((pred - actual) ** 2)
    return {name: statistics.mean(e) for name, e in errs.items()}
def select_model(mse_by_model, n_days, total_spend):
    if not mse_by_model or "naive7" not in mse_by_model:
        return "naive7"
    if n_days < MIN_HISTORY_FOR_SELECTION or total_spend < 1.0:
        return "naive7"
    naive_mse = mse_by_model["naive7"]
    best_name, best_mse = "naive7", naive_mse
    for name, mse in mse_by_model.items():
        if name == "naive7":
            continue
        if mse < best_mse and mse < naive_mse * 0.95:
            best_name, best_mse = name, mse
    return best_name
def train_predict_cost(values, dates):
    n = min(len(values), len(dates))
    if n == 0:
        return 0.0
    values, dates = values[:n], dates[:n]
    total_spend = sum(values)
    if total_spend <= 0:
        return 0.0
    origins = rolling_origins(n)
    mse_by_model = backtest_service(values, dates, origins)
    chosen = select_model(mse_by_model, n, total_spend)
    return round(CANDIDATES.get(chosen, f_naive7)(values, dates, n), 2)
def forecast_costs(daily_history, dates):
    return {label: train_predict_cost(values, dates) for label, values in daily_history.items()}

# ==================== Anomaly Detection ====================
def build_cost_features(values):
    features = []
    for i, current in enumerate(values):
        avg7 = statistics.mean(values[max(0, i - 6):i + 1])
        avg30 = statistics.mean(values[max(0, i - 29):i + 1])
        pct_change = ((current - avg30) / avg30 * 100) if avg30 else 0.0
        prev = values[i - 1] if i > 0 else current
        features.append([current, avg7, avg30, pct_change, prev])
    return features
def detect_anomalies():
    end = date.today() + timedelta(days=1)
    start = end - timedelta(days=ANOMALY_WINDOW_DAYS)
    resp = ce_query(start, end, granularity="DAILY", group_by="SERVICE")
    if not resp:
        return []
    series = {}
    for day in resp["ResultsByTime"]:
        for g in day["Groups"]:
            series.setdefault(g["Keys"][0], []).append(float(g["Metrics"]["UnblendedCost"]["Amount"]))
    anomalies = []
    for name, values in series.items():
        if len(values) < ANOMALY_MIN_HISTORY_DAYS:
            continue
        latest = values[-1]
        if latest < ANOMALY_MIN_COST:
            continue
        features = build_cost_features(values)
        X = np.array(features)
        model = IsolationForest(n_estimators=100, contamination=ANOMALY_CONTAMINATION, random_state=42)
        model.fit(X)
        if model.predict(X)[-1] != -1:
            continue
        avg30, pct_change = features[-1][2], features[-1][3]
        if avg30 <= 0 or pct_change < ANOMALY_SPIKE_THRESHOLD_PCT:
            continue
        score = float(model.decision_function(X)[-1])
        explanation = f"Today's {name} cost (${latest:.2f}) is {pct_change:.0f}% above its {ANOMALY_WINDOW_DAYS}-day average of ${avg30:.2f}."
        anomalies.append({"service": name, "latest": latest, "score": score, "pct_increase": pct_change, "explanation": explanation})
    return sorted(anomalies, key=lambda a: a["score"])

# ==================== Rightsizing ====================
def get_low_cpu_instances():
    try:
        paginator = ec2.get_paginator("describe_instances")
        pages = paginator.paginate(Filters=[{"Name": "instance-state-name", "Values": ["running"]}])
        instances = [inst for page in pages for r in page["Reservations"] for inst in r["Instances"]]
    except ClientError as e:
        print(f"describe_instances error: {e}")
        return []
    end = datetime.utcnow()
    start = end - timedelta(days=CPU_LOOKBACK_DAYS)
    candidates = []
    for inst in instances:
        instance_id = inst["InstanceId"]
        try:
            metrics = cloudwatch.get_metric_statistics(
                Namespace="AWS/EC2", MetricName="CPUUtilization",
                Dimensions=[{"Name": "InstanceId", "Value": instance_id}],
                StartTime=start, EndTime=end, Period=86400, Statistics=["Average"])
            points = [p["Average"] for p in metrics["Datapoints"]]
        except ClientError as e:
            print(f"get_metric_statistics error for {instance_id}: {e}")
            continue
        if not points:
            continue
        avg_cpu = statistics.mean(points)
        if avg_cpu < LOW_CPU_THRESHOLD:
            candidates.append({"instance_id": instance_id, "instance_type": inst["InstanceType"], "avg_cpu": avg_cpu})
    return candidates

# ==================== Idle Resources ====================
def get_idle_ebs_volumes():
    try:
        paginator = ec2.get_paginator("describe_volumes")
        pages = paginator.paginate(Filters=[{"Name": "status", "Values": ["available"]}])
        return [{"volume_id": v["VolumeId"], "size_gb": v["Size"]} for page in pages for v in page["Volumes"]]
    except ClientError as e:
        print(f"describe_volumes error: {e}")
        return []
def get_all_volumes_info():
    try:
        paginator = ec2.get_paginator("describe_volumes")
        volumes = [v for page in paginator.paginate() for v in page["Volumes"]]
        total_gb = sum(v["Size"] for v in volumes)
        volume_ids = {v["VolumeId"] for v in volumes}
        return total_gb, volume_ids
    except ClientError as e:
        print(f"describe_volumes (all) error: {e}")
        return 0, set()
def get_unused_eips():
    try:
        addresses = ec2.describe_addresses()["Addresses"]
        return [{"public_ip": a.get("PublicIp", "N/A")} for a in addresses
                if "AssociationId" not in a and "InstanceId" not in a]
    except ClientError as e:
        print(f"describe_addresses error: {e}")
        return []
def get_all_snapshot_gb():
    try:
        paginator = ec2.get_paginator("describe_snapshots")
        total = 0
        for page in paginator.paginate(OwnerIds=["self"]):
            for s in page["Snapshots"]:
                total += s.get("VolumeSize", 0)
        return total
    except ClientError as e:
        print(f"describe_snapshots (all) error: {e}")
        return 0
def get_old_snapshots(existing_volume_ids):
    try:
        paginator = ec2.get_paginator("describe_snapshots")
        cutoff = date.today() - timedelta(days=OLD_SNAPSHOT_DAYS)
        flagged = []
        for page in paginator.paginate(OwnerIds=["self"]):
            for s in page["Snapshots"]:
                age_days = (date.today() - s["StartTime"].date()).days
                is_old = s["StartTime"].date() < cutoff
                is_orphaned = s.get("VolumeId") not in existing_volume_ids
                if is_old or is_orphaned:
                    reasons = (["source volume gone"] if is_orphaned else []) + ([f"{age_days}d old"] if is_old else [])
                    flagged.append({"snapshot_id": s["SnapshotId"], "age_days": age_days,
                                     "size_gb": s.get("VolumeSize", 0), "reason": ", ".join(reasons)})
        return flagged
    except ClientError as e:
        print(f"describe_snapshots error: {e}")
        return []

# ==================== Report Builder ====================
TH = "background:#1f4e79;color:#fff;text-align:left;padding:8px 10px;font-size:13px;"
TD = "padding:7px 10px;border-bottom:1px solid #e3e6ea;font-size:13px;"
TD_ALT = TD + "background:#f7f9fc;"
TABLE = "border-collapse:collapse;width:100%;margin:0 0 22px 0;font-family:Arial,Helvetica,sans-serif;"
def html_table(headers, rows):
    thead = "".join(f'<th style="{TH}">{h}</th>' for h in headers)
    body = "".join(
        "<tr>" + "".join(f'<td style="{TD_ALT if i % 2 else TD}">{c}</td>' for c in row) + "</tr>"
        for i, row in enumerate(rows)
    )
    return f'<table style="{TABLE}"><thead><tr>{thead}</tr></thead><tbody>{body}</tbody></table>'
def section(title, body_html):
    heading = f'<h2 style="color:#1f4e79;font-family:Arial,sans-serif;font-size:16px;border-bottom:2px solid #1f4e79;padding-bottom:5px;margin:26px 0 12px;">{title}</h2>'
    return heading + body_html
def pct_color(change):
    return "#c0392b" if change >= 0 else "#2e7d32"
def build_html_report(total_cost, service_rows, n_days, top_svc, daily_history, cost_forecast, mom_anomalies, anomalies,
                       rightsizing, idle_ebs, unused_eips, old_snapshots):
    p = 'font-family:Arial,sans-serif;font-size:13px;color:#333;margin:0 0 10px;'
    top_rows = [[name, f"${cost:,.2f}"] for name, cost in top_svc]
    top_html = section("Top Cost-Consuming Services", html_table(["Service", "Current Month"], top_rows))
    cmp_rows = [[label, f"${cur:,.2f}", f"${prev:,.2f}"] for label, cur, prev in service_rows]
    cmp_caption = f'<p style="{p}">Comparing the first {n_days} day(s) of this month against the first {n_days} day(s) of last month.</p>'
    cmp_html = section("Service Cost Comparison",
                        cmp_caption + html_table(["Service", f"Current (First {n_days}d)", f"Previous (First {n_days}d)"], cmp_rows))
    if mom_anomalies:
        mom_rows = [[a["label"], f"${a['prev']:,.2f}", f"${a['cur']:,.2f}",
                     f'<span style="color:#c0392b;font-weight:bold;">{a["change_pct"]:+.1f}%</span>'] for a in mom_anomalies]
        mom_body = html_table(["Service", "Previous", "Current", "Change %"], mom_rows)
        mom_body += f'<p style="{p}"><strong>Recommendation:</strong> Review recent config/usage changes for the services above.</p>'
    else:
        mom_body = f'<p style="{p}">No service increased by {MOM_INCREASE_THRESHOLD_PCT:.0f}% or more versus the comparable period last month.</p>'
    mom_html = section("Abnormal Cost Increases (Same-Period Comparison)", mom_body)
    fc_rows = []
    for label, values in daily_history.items():
        last_7 = sum(values[-7:]) if len(values) >= 7 else sum(values)
        pred_7 = cost_forecast.get(label, 0.0)
        change_7_pct = ((pred_7 - last_7) / last_7 * 100) if last_7 > 0 else 0.0
        fc_rows.append([label, f"${last_7:,.2f}", f"${pred_7:,.2f}",
                         f'<span style="color:{pct_color(change_7_pct)};font-weight:bold;">{change_7_pct:+.1f}%</span>'])
    fc_html = section("Cost Forecast (Next 7 Days)",
                       html_table(["Service", "Last 7 Days", "Predicted 7 Days", "7d Change"], fc_rows))
    if anomalies:
        cards = "".join(
            f'<div style="border-left:4px solid #c0392b;background:#fdecea;padding:10px 14px;margin-bottom:10px;{p}">'
            f'<strong>{a["service"]}</strong> — Latest: ${a["latest"]:.2f} | Anomaly Score: {a["score"]:.3f} | '
            f'Increase: +{a["pct_increase"]:.0f}%<br><em>{a["explanation"]}</em></div>'
            for a in anomalies
        )
    else:
        cards = f'<p style="{p}">No significant cost increase anomalies detected.</p>'
    anomaly_html = section("Cost Anomalies (Spikes Only)", cards)
    if rightsizing:
        r_rows = [[r["instance_id"], r["instance_type"], f'{r["avg_cpu"]:.1f}%'] for r in rightsizing]
        r_body = html_table(["Instance ID", "Type", "Avg CPU %"], r_rows)
        r_body += f'<p style="{p}"><strong>Recommendation:</strong> Consider downsizing or stopping these instances after review.</p>'
    else:
        r_body = f'<p style="{p}">No low-CPU instances found.</p>'
    right_html = section("Rightsizing Candidates (Low CPU EC2 Instances)", r_body)
    idle_rows = [["EBS Volume", v["volume_id"], f'{v["size_gb"]} GB', f'${v.get("est_monthly_cost", 0.0):,.2f}', "Delete if unused"] for v in idle_ebs]
    idle_rows += [["Elastic IP", e["public_ip"], "Unassociated", "-", "Release if unused"] for e in unused_eips]
    SNAPSHOT_DISPLAY_LIMIT = 3
    idle_rows += [["Snapshot", s["snapshot_id"], s["reason"], f'${s.get("est_monthly_cost", 0.0):,.2f}', "Review/Delete"]
                  for s in old_snapshots[:SNAPSHOT_DISPLAY_LIMIT]]
    idle_body = html_table(["Type", "Resource ID", "Detail", "Est. $/mo", "Recommendation"], idle_rows)
    if len(old_snapshots) > SNAPSHOT_DISPLAY_LIMIT:
        remaining = len(old_snapshots) - SNAPSHOT_DISPLAY_LIMIT
        remaining_cost = sum(s.get("est_monthly_cost", 0.0) for s in old_snapshots[SNAPSHOT_DISPLAY_LIMIT:])
        idle_body += f'<p style="{p}">... and {remaining} more snapshot(s) flagged (est. ${remaining_cost:,.2f}/mo). See Optimization Summary for total count.</p>'
    idle_html = section("Idle Resources", idle_body)
    gov_html = section("Cost Governance Recommendations", f'''<ul style="{p}padding-left:20px;">
        <li>Configure AWS Budgets with service-level and account-level thresholds.</li>
        <li>Enable AWS Cost Anomaly Detection for proactive notifications.</li>
        <li>Configure Amazon CloudWatch billing alarms.</li>
        <li>Review usage trends and recent service/config changes to confirm the root cause of any flagged spikes above.</li>
    </ul>''')

    summary_html = section("Optimization Summary", f'''
        <div style="background:#f0f6ff;border-radius:6px;padding:14px 18px;{p}">
        <ul style="padding-left:20px;margin:0;">
        <li>{len(rightsizing)} EC2 instance(s) identified for rightsizing review.</li>
        <li>{len(idle_ebs)} unattached EBS volume(s) detected.</li>
        <li>{len(unused_eips)} unused Elastic IP(s) detected.</li>
        <li>{len(old_snapshots)} snapshot(s) flagged as old/orphaned.</li>
        </ul></div>''')

    return f'''<html><body style="background:#eef1f5;margin:0;padding:24px;">
    <div style="max-width:760px;margin:0 auto;background:#ffffff;padding:28px 32px;border-radius:8px;font-family:Arial,sans-serif;">
      <h1 style="color:#1f4e79;margin:0 0 4px;font-size:22px;">AWS Cost Report</h1>
      <p style="color:#888;margin:0;font-size:12px;">Generated on {date.today().isoformat()}</p>
      <hr style="border:none;border-top:1px solid #1f4e79;opacity:0.4;margin:16px 0 20px;" />
      <div style="background:#1f4e79;color:#fff;padding:16px 20px;border-radius:6px;margin-bottom:10px;">
        <div style="font-size:13px;opacity:0.85;">Total Monthly Cost (Month to Date)</div>
        <div style="font-size:30px;font-weight:bold;">${total_cost:,.2f}</div>
      </div>
      {top_html}{cmp_html}{mom_html}{fc_html}{anomaly_html}{right_html}{idle_html}{gov_html}{summary_html}
      <p style="color:#aaa;font-size:11px;margin-top:30px;">Automated report from AWS Cost Report Lambda.</p>
    </div></body></html>'''
def build_text_summary(total_cost):
    return (f"AWS Cost Report - Total Monthly Cost (Month to Date): ${total_cost:,.2f}\n"
            f"View this email in an HTML-capable client to see the full report.")
def send_report_email(html_report, text_summary):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = "AWS Cost Report"
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL
    msg.attach(MIMEText(text_summary, "plain", "utf-8"))
    msg.attach(MIMEText(html_report, "html", "utf-8"))
    return ses.send_raw_email(
        Source=SENDER_EMAIL,
        Destinations=[RECEIVER_EMAIL],
        RawMessage={"Data": msg.as_bytes()},
    )

# ==================== Lambda Handler ====================
def lambda_handler(event, context):
    cur_start, cur_end, prev_start, prev_end = month_bounds()
    total_cost = get_total_cost(cur_start, cur_end)
    service_rows, cur_costs, n_days = build_service_comparison(cur_start, cur_end, prev_start, prev_end)
    top_svc = top_services(cur_costs)
    mom_anomalies = get_mom_anomalies(service_rows)
    daily_history, history_dates = get_service_daily_history()
    cost_forecast = forecast_costs(daily_history, history_dates)
    anomalies = detect_anomalies()
    rightsizing = get_low_cpu_instances()
    total_ebs_gb, existing_volume_ids = get_all_volumes_info()
    idle_ebs = get_idle_ebs_volumes()
    ebs_storage_cost = get_usage_type_cost(cur_start, cur_end, "VolumeUsage")
    ebs_per_gb_rate = (ebs_storage_cost / total_ebs_gb) if total_ebs_gb else 0.0
    for v in idle_ebs:
        v["est_monthly_cost"] = project_to_full_month(ebs_per_gb_rate * v["size_gb"], cur_start)
    unused_eips = get_unused_eips()
    total_snapshot_gb = get_all_snapshot_gb()
    snapshot_storage_cost = get_usage_type_cost(cur_start, cur_end, "EBS:SnapshotUsage")
    snapshot_per_gb_rate = (snapshot_storage_cost / total_snapshot_gb) if total_snapshot_gb else 0.0
    old_snapshots = get_old_snapshots(existing_volume_ids)
    for s in old_snapshots:
        s["est_monthly_cost"] = project_to_full_month(snapshot_per_gb_rate * s["size_gb"], cur_start)
    html_report = build_html_report(total_cost, service_rows, n_days, top_svc, daily_history, cost_forecast, mom_anomalies,
                                     anomalies, rightsizing, idle_ebs, unused_eips, old_snapshots)
    email_sent, email_error, message_id = False, None, None
    try:
        response = send_report_email(html_report, build_text_summary(total_cost))
        message_id = response.get("MessageId")
        email_sent = True
        print(f"ses.send_raw_email response: {response}")
        print(f"ses.send_raw_email MessageId: {message_id}")
    except ClientError as e:
        email_error = str(e.response.get("Error", e.response))
        print(f"ses.send_raw_email ClientError: {e.response}")
    except Exception as e:
        email_error = str(e)
        print(f"ses.send_raw_email unexpected error: {e}")
    return {"statusCode": 200 if email_sent else 500, "email_sent": email_sent, "email_error": email_error,
            "message_id": message_id, "total_monthly_cost": total_cost, "mom_anomalies_found": len(mom_anomalies),
            "anomalies_found": len(anomalies), "rightsizing_candidates": len(rightsizing),
            "idle_resources": len(idle_ebs) + len(unused_eips) + len(old_snapshots)}