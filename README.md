**1. CPU Anomaly Detection**

The CPU Anomaly Detection project is an AWS-based monitoring system that detects unusual CPU utilization in EC2 instances using Amazon CloudWatch and the Isolation Forest machine learning algorithm. The system analyzes CPU usage metrics, identifies abnormal patterns, and uses AWS Lambda to process the results and Amazon SNS to send alerts when an anomaly is detected. This helps in proactive monitoring and prevents potential performance issues.

**2. VPC Network Traffic Anomaly Detection**

The VPC Network Traffic Anomaly Detection project is an AWS-based security solution that analyzes VPC Flow Logs to identify suspicious or unusual network traffic. The system collects flow-log data through CloudWatch Logs, and AWS Lambda uses the Isolation Forest algorithm to analyze features such as source/destination ports, protocols, packets, and bytes. When anomalous traffic is detected, the system identifies suspicious source IPs and sends an alert through Amazon SNS.

**3. ML-Based FinOps Cost Optimization and Governance**

The ML-Based FinOps Cost Optimization and Governance project is an AWS-based cloud cost management solution that uses machine learning and AWS services to monitor, analyze, forecast, and optimize cloud expenditure. The system uses AWS Cost Explorer to analyze account-level and service-level costs, performs same-period cost comparisons to identify abnormal increases, and applies the Isolation Forest algorithm to detect unusual cost spikes. For cost forecasting, multiple time-series forecasting approaches are evaluated using rolling-origin backtesting, and the best-performing model is selected to predict costs for the next seven days.

The system also analyzes EC2 utilization using Amazon CloudWatch to identify potential rightsizing candidates and detects unused or potentially unnecessary resources such as unattached EBS volumes, unassociated Elastic IP addresses, and old or orphaned EBS snapshots. AWS Lambda performs the analysis and generates an automated HTML cost report containing cost trends, anomalies, forecasts, optimization recommendations, and governance suggestions. Amazon SES is used to deliver the generated report through email. The solution helps organizations improve cloud cost visibility, proactively identify cost anomalies, forecast future expenditure, and support FinOps-based cost governance and optimization.
