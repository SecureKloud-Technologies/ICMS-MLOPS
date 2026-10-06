**1. CPU Anomaly Detection**

The CPU Anomaly Detection project is an AWS-based monitoring system that detects unusual CPU utilization in EC2 instances using Amazon CloudWatch and the Isolation Forest machine learning algorithm. The system analyzes CPU usage metrics, identifies abnormal patterns, and uses AWS Lambda to process the results and Amazon SNS to send alerts when an anomaly is detected. This helps in proactive monitoring and prevents potential performance issues.

**2. VPC Network Traffic Anomaly Detection **

The VPC Network Traffic Anomaly Detection project is an AWS-based security solution that analyzes VPC Flow Logs to identify suspicious or unusual network traffic. The system collects flow-log data through CloudWatch Logs, and AWS Lambda uses the Isolation Forest algorithm to analyze features such as source/destination ports, protocols, packets, and bytes. When anomalous traffic is detected, the system identifies suspicious source IPs and sends an alert through Amazon SNS.
