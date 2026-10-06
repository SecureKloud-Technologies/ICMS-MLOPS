import os
import boto3
import pandas as pd
from sklearn.ensemble import IsolationForest

# AWS Clients
logs = boto3.client("logs")
sns = boto3.client("sns")

# Environment Variables
SNS_TOPIC_ARN = os.environ["SNS_TOPIC_ARN"]
LOG_GROUP = os.environ["LOG_GROUP"]

# Protocol Mapping
PROTOCOLS = {
    1: "ICMP",
    6: "TCP",
    17: "UDP",
    47: "GRE",
    50: "ESP",
    51: "AH"
}


def lambda_handler(event, context):

    # Get all log streams
    streams = logs.describe_log_streams(
        logGroupName=LOG_GROUP
    )

    flow_logs = []

    # Read flow logs
    for stream in streams["logStreams"]:

        events = logs.get_log_events(
            logGroupName=LOG_GROUP,
            logStreamName=stream["logStreamName"],
            limit=100
        )

        for event in events["events"]:

            fields = event["message"].split()

            if len(fields) < 14:
                continue

            try:
                flow_logs.append({
                    "src_ip": fields[3],
                    "dst_ip": fields[4],
                    "src_port": int(fields[5]),
                    "dst_port": int(fields[6]),
                    "protocol": int(fields[7]),
                    "packets": int(fields[8]),
                    "bytes": int(fields[9]),
                    "action": fields[12]
                })

            except Exception:
                continue

    # Convert to DataFrame
    df = pd.DataFrame(flow_logs)

    if df.empty:
        return {
            "statusCode": 404,
            "message": "No VPC Flow Log data found."
        }

    # Features used for Machine Learning
    features = df[
        [
            "src_port",
            "dst_port",
            "protocol",
            "packets",
            "bytes"
        ]
    ]

    # Isolation Forest Model
    model = IsolationForest(
        contamination="auto",
        random_state=42
    )

    model.fit(features)

    df["prediction"] = model.predict(features)

    df["anomaly_score"] = model.decision_function(features)

    df["status"] = df["prediction"].map({
        1: "Normal",
        -1: "Anomaly"
    })

    # Filter anomalies
    anomalies = df[df["prediction"] == -1].copy()

    # Convert protocol number to name
    anomalies["protocol_name"] = anomalies["protocol"].apply(
        lambda x: PROTOCOLS.get(x, str(x))
    )

    # Group anomalies by Source IP
    suspicious_ips = (
        anomalies
        .groupby("src_ip")
        .agg(
            anomaly_count=("src_ip", "count"),
            total_packets=("packets", "sum"),
            total_bytes=("bytes", "sum"),
            protocol=("protocol_name", "first"),
            anomaly_score=("anomaly_score", "min"),
            action=("action", "first")
        )
        .reset_index()
        .sort_values(
            by="anomaly_count",
            ascending=False
        )
    )

    # Prepare SNS Message
    if not anomalies.empty:

        message = (
            f"VPC Network Traffic Anomaly Detected\n\n"
            f"Total Records Analyzed : {len(df)}\n"
            f"Total Anomalies        : {len(anomalies)}\n"
            f"Unique Suspicious IPs  : {len(suspicious_ips)}\n\n"
        )

        for _, row in suspicious_ips.iterrows():

            message += (
                f"Source IP            : {row['src_ip']}\n"
                f"Total Anomalous Flows: {row['anomaly_count']}\n"
                f"Lowest Anomaly Score : {row['anomaly_score']:.4f}\n\n"
            )

            ip_anomalies = anomalies[
                anomalies["src_ip"] == row["src_ip"]
            ]

            flow_no = 1

            for _, anomaly in ip_anomalies.iterrows():

                message += (
                    f"Flow {flow_no}\n"
                    f"Destination IP   : {anomaly['dst_ip']}\n"
                    f"Source Port      : {anomaly['src_port']}\n"
                    f"Destination Port : {anomaly['dst_port']}\n"
                    f"Protocol         : {anomaly['protocol_name']}\n"
                    f"Packets          : {anomaly['packets']}\n"
                    f"Bytes            : {anomaly['bytes']}\n"
                    f"Action           : {anomaly['action']}\n"
                    f"Anomaly Score    : {anomaly['anomaly_score']:.4f}\n"
                    f"{'-' * 40}\n"
                )

                flow_no += 1

            message += "=" * 60 + "\n\n"

        sns.publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject="VPC Network Traffic Anomaly Detected",
            Message=message
        )

    return {
        "statusCode": 200,
        "total_records_analyzed": len(df),
        "total_anomalies": len(anomalies),
        "unique_suspicious_ips": len(suspicious_ips),
        "top_suspicious_ips": suspicious_ips.to_dict(
            orient="records"
        )
    }
