#######################################################################################################################################################
# 
# Name: Aakriti Goyal
# SID: 740099010
# Exam Date: 28/03/2025
# Module: BEMM458 (Programming for Business analytics)
# Github link for this assignment:   
#
# ######################################################################################################################################################
# Instruction 1. Read the questions and instructions carefully and complete scripts.

# Instruction 2. Only ethical and minimal use of AI is allowed. You might use AI to give advice on how to use a tool or programming language.  
#                You must not use AI to create the code. You might make use of AI to aid debugging of the code.  
#                You must indicate clearly how and where you have used AI.

# Instruction 3. Copy the output of the code and insert it as a comment below your code e.g # OUTPUT (23,45)

# Instruction 4. Ensure you provide comments for the code and the output to show contextual understanding.

# Instruction 5. Upon completing this test, commit to Git, copy and paste your code into the word file and upload the saved file to ELE.
#                There is a leeway on when you need to upload to ELE, however, you must commit to Git at 
#                the end of your session.

# ######################################################################################################################################################
# Question 1 - Data Processing and Loops
# You are given a string representing customer reviews. Use a for loop to process the text and count occurrences of specific keywords.
# Your allocated keywords are determined by the first and last digit of your SID.
# Count and store occurrences of each keyword in a dictionary called keyword_counts.
 
customer_reviews = """The product is well-designed and user-friendly. However, I experienced some issues with durability. The customer service was helpful, 
but I expected a faster response. The quality of the materials used is excellent. Overall, the purchase was satisfactory."""

# Keywords dictionary
keywords = {
    0: 'user-friendly',
    1: 'helpful',
    2: 'durability',
    3: 'response',
    4: 'satisfactory',
    5: 'quality',
    6: 'service',
    7: 'issues',
    8: 'purchase',
    9: 'materials'
}
import re
# Write your code to process the text and count keyword occurrences
SID_first_digit = 7
SID_last_digit = 0

allocated_keywords = [keywords[SID_first_digit], keywords[SID_last_digit]]

keyword_counts = {
   keyword: 0 for keyword in allocated_keywords
}

for keyword in allocated_keywords:
    keyword_counts[keyword] = len(re.findall(rf'\b{keyword}\b', customer_reviews, re.IGNORECASE))
print("The output dictionary : ",keyword_counts)
##########################################################################################################################################################

# Question 2 - Business Metrics
# Scenario - You work in an online retail company as a business analyst. Your manager wants weekly reports on financial performance, including:
# Gross Profit Margin, Inventory Turnover, Customer Retention Rate (CRR), and Break-even Analysis. Implement Python functions 
# that take relevant values as inputs and return the computed metric. Use the first two and last two digits of your ID number as input values.

# Insert first two digits of ID number here:
SID_first_two_digits = 74

# Insert last two digits of ID number here:
SID_last_two_digits = 10

# Write your function for Gross Profit Margin
def gpm(revenue, cogs):
    if revenue == 0:
        return 0
    return ((revenue - cogs) / revenue) * 100


# Write your function for Inventory Turnover
def ivt(cost_of_goods_sold, avg_inventory):
    if avg_inventory == 0:
        return 0
    return cost_of_goods_sold / avg_inventory

# Write your function for Customer Retention Rate (CRR)
def crr(initial_customers, retained_customers):
    if initial_customers == 0:
        return 0
    return (retained_customers / initial_customers) * 100

# Write your function for Break-even Analysis
def bep(fixed_costs, price_per_unit, variable_cost_per_unit):
    if price_per_unit - variable_cost_per_unit == 0:
        return 0
    return fixed_costs / (price_per_unit - variable_cost_per_unit)

# Call your functions here


print("Gross Profit Margin is: ", gpm(SID_first_two_digits*1000,SID_last_two_digits*500))
print("The Inventory Turnover is: ", ivt(SID_first_two_digits*500,SID_last_two_digits*200))
print("The Customer Retention Rate is: ", crr(SID_first_two_digits*10,SID_last_two_digits*2))
print("Break-even point: ", bep(SID_first_two_digits * 5000, SID_last_two_digits * 100, SID_first_two_digits * 20))

##########################################################################################################################################################

# Question 3 - Forecasting and Regression
# A logistics company has gathered data on delivery costs and shipment volumes. The table below provides different costs and their corresponding shipment volumes.
# Develop a linear regression model and determine:
# 1. The optimal delivery cost that maximizes profit
# 2. The expected shipment volume when the cost is set at £68

"""
Delivery Cost (£)    Shipment Volume (Units)
-------------------------------------------
25                  500
30                  480
35                  450
40                  420
45                  400
50                  370
55                  340
60                  310
65                  290
70                  250
"""

# Write your regression model code here
#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Data 
costs = np.array([25, 30, 35, 40, 45, 50, 55, 60, 65, 70]).reshape(-1, 1)
volumes = np.array([500, 480, 450, 420, 400, 370, 340, 310, 290, 250])

# Fitting linear regression model
model = LinearRegression()
model.fit(costs, volumes)

# Predict shipment volume for cost = 68
cost_68 = np.array([[68]])
predicted_volume_68 = model.predict(cost_68)[0]
print(f"Predicted shipment volume at £68: {predicted_volume_68:.2f} units")

# Finding the optimal delivery cost to maximize profit
# Assume profit = (Revenue - Cost), where Revenue = Shipment Volume * Cost
cost_range = np.arange(25, 71, 1).reshape(-1, 1)
predicted_volumes = model.predict(cost_range)
profits = (cost_range.flatten() * predicted_volumes) - (cost_range.flatten() * 100)  # Assuming a fixed cost per unit
optimal_index = np.argmax(profits)
optimal_cost = cost_range[optimal_index][0]

print(f"Optimal delivery cost to maximize profit: £{optimal_cost}")

# Plotting the regression model results
plt.scatter(costs, volumes, color='blue', label='Actual Data')
plt.plot(cost_range, predicted_volumes, color='red', linestyle='--', label='Regression Line')
plt.xlabel("Delivery Cost (£)")
plt.ylabel("Shipment Volume (Units)")
plt.title("Delivery Cost vs Shipment Volume")
plt.legend()
plt.show()

##########################################################################################################################################################

# Question 4 - Debugging and Data Visualization

import random
import matplotlib.pyplot as plt

# Get student ID and generate random numbers
your_ID = input("Enter your Student ID: ")
max_value = int(your_ID)  # Convert input to integer

# Generate 100 random numbers between 1 and student ID number
random_numbers = [random.randint(1, max_value) for _ in range(100)]

# Plot the histogram
plt.hist(random_numbers, bins=10, edgecolor='blue', alpha=0.7, color='red')
plt.title("Histogram of 100 Random Numbers")
plt.xlabel("Value Range")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()


