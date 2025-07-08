# Cloud-Driven Sustainability: Carbon Footprint Estimation and Environmental Impact Analysis 
![bg](https://github.com/user-attachments/assets/925ccc4b-80d3-4985-b153-5ad3d45403be)
Amazon Web Services (AWS) based project to store, process, and analyze individual lifestyle data.

## Introduction
Climate change and global warming, driven largely by carbon emissions, present an urgent global challenge that demands immediate action. With individuals contributing significantly through daily activities like energy use, dietary choices, transportation, and waste disposal, there is a pressing need for accessible tools that provide accurate carbon footprint estimates and practical recommendations to promote sustainable behavior. This work introduces a cloud-based system on AWS that leverages machine learning to estimate individual carbon footprints based on lifestyle data and deliver actionable insights to reduce emissions. The system utilizes scalable cloud infrastructure to ensure global accessibility, real-time processing, and cost-effectiveness while maintaining energy efficiency and ethical operational standards. Through dynamic visualizations, tailored recommendations, and a user-friendly interface, the solution not only empowers individuals but also serves as a valuable resource for policymakers, educators, and organizations. This approach bridges the gap between awareness and action, fostering a deeper understanding of environmental impact and enabling collective progress toward sustainability goals

## Contributions
•	Carbon Footprint Estimation
•	Cloud-Based Implementation
•	Personalized Recommendations
•	Interactive Visualizations

The work focuses on leveraging AWS services to build a carbon footprint estimation and recommendation system. Amazon S3 serves as the data storage for the lifestyle sustainability dataset, along with other web app related files. The Flask app was developed using AWS Cloud9, a cloud based IDE (Integrated Development Environment) that allows for the design, running and debugging of code, and secure coding companion, and the basic functionality for serving web pages and processing user inputs has been successfully implemented. 
A static website has been hosted on Amazon S3, providing the front-end for user interactions. The initial models for carbon footprint estimation have been trained on EC2 and stored in S3 for future deployment.
The Docker image for the Flask app has been built and successfully pushed to Amazon ECR, enabling deployment on AWS Fargate. ECS clusters, task definitions, services, and security groups were set up to manage the Flask app’s deployment and scaling, ensuring smooth API interactions.
Postman was used to test the Flask API, while Python on EC2 was used to perform data analysis and visualizations for analyzing carbon footprint data.

## Methodology
Data Storage ->	Data Processing -> Model Training -> Model Deployment -> Recommendation Generation -> Web Application Development 
-> Data Analysis and Visualization ->	Documentation
![system design](https://github.com/user-attachments/assets/d0ff1650-c007-488d-9bea-5271361d33db)
Fig. System Design for the Cloud Based Framework

This system design employs various AWS services such as Cloud9, S3, ECR, ECS, Fargate, EC2, CloudWatch, along with Docker images, ensuring a highly scalable, server-less, and cost-effective solution. The use of server-less technologies like Fargate and S3 ensures that the application scales automatically based on user demands, while services like CloudWatch offer comprehensive monitoring and logging. This architecture is robust, cloud-native, and optimally designed for large-scale deployment with minimal operational overhead.

### Models tried in this work 
#### Machine Learning Models 
Linear Regression, Random Forest, XGBoost, Gradient Boosting, CatBoost, LightGBM, SVR, ElasticNet, AdaBoost, Decision Tree
#### Deep Learning Models 
Deep Neural Network (DNN), Convolutional Neural Network (CNN), Long Short-Term Memory Model (LSTM)


### Dataset <a href="https://www.kaggle.com/datasets/naveennas/sustainable-lifestyle-rating-dataset/data" target="_blank" title="Dataset">   <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=flat&logo=kaggle&logoColor=white" alt="Kaggle"/> </a>


### Results
The Gradient Boosting Regressor model achieved an R² score of 0.9871 and a MSE of 99.32. The model showed excellent predictive accuracy, suggesting a very high correlation between the selected features and the estimated carbon emissions. EnergySource is the most influential factor (29.7% importance) in determining an individual’s carbon footprint, followed by DietType (19.39%) and MonthlyElectricityConsumption (15.63%). This aligns with real-world data, where energy consumption, particularly non-renewable energy, is a significant contributor to carbon emissions, and dietary choices play a substantial role in overall environmental impact.



