import boto3
import pandas as pd
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta, timezone
cloudwatch = boto3.client("cloudwatch")
ec2 = boto3.client("ec2")
sns = boto3.client("sns")
TOPIC_ARN = "arn:aws:sns:ap-south-1:466749146115:cpu-anomaly-alerts"
def lambda_handler(event, context):
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=1)
    # Get all running EC2 instances
    response = ec2.describe_instances(
        Filters=[
            {
                "Name": "instance-state-name",
                "Values": ["running"]
            }
        ]
    )
    instances = []
    for reservation in response["Reservations"]:
        for instance in reservation["Instances"]:
            instances.append(instance["InstanceId"])
    if not instances:
        return {
            "statusCode": 404,
            "body": "No running EC2 instances found."
        }
    results = []
    for instance_id in instances:
        metric_response = cloudwatch.get_metric_statistics(
            Namespace="AWS/EC2",
            MetricName="CPUUtilization",
            Dimensions=[
                {
                    "Name": "InstanceId",
                    "Value": instance_id
                }
            ],
            StartTime=start_time,
            EndTime=end_time,
            Period=300,
            Statistics=["Average"]
        )
        datapoints = metric_response["Datapoints"]
        if not datapoints:
            continue
        df = pd.DataFrame(datapoints)
        df = df.sort_values("Timestamp")
        cpu = df[["Average"]]
        model = IsolationForest(
            contamination=0.05,
            random_state=42
        )
        df["Prediction"] = model.fit_predict(cpu)
        df["Status"] = df["Prediction"].map({
            1: "Normal",
            -1: "Anomaly"
        })
        anomalies = df[df["Prediction"] == -1]
        df["Timestamp"] = df["Timestamp"].astype(str)
        anomalies["Timestamp"] = anomalies["Timestamp"].astype(str)
        if len(anomalies) > 0:
            report = df[["Timestamp", "Average", "Status"]].to_string(index=False)
            message = f"""
CPU Anomaly Report
Instance ID: {instance_id}
Total Data Points: {len(df)}
Anomalies Detected: {len(anomalies)}
=========================================
All CPU Readings
{report}
=========================================

Anomaly Details
{anomalies[['Timestamp','Average']].to_string(index=False)}
"""
            sns.publish(
                TopicArn=TOPIC_ARN,
                Subject=f"CPU Anomaly Alert - {instance_id}",
                Message=message
            )
        results.append({
            "instance_id": instance_id,
            "total_points": len(df),
            "anomalies_detected": len(anomalies)
        })
    return {
        "statusCode": 200,
        "body": results
    }
