# [NBAScorePredictorML]

## Background / Inspiration for the Project
ML model for predicting NBA player and team points, leveraging advanced analytics for game and performance insights. Ideal for fans, analysts, and fantasy leagues seeking data-driven predictions.

## Use Cases
- Sports analysts looking for insights into player and team performance in the NBA.
- News Broadcasts looking to promote games that are predicted to be very close in point totals or possibly even games where there could be big upsets.
- Fantasy sports enthusiasts seeking data-backed predictions, on who to start, sit, trade, and/or acquire from waivers.
- Companies specializing in sports betting or predictive modeling for sports events.

## Model Performance and Insights

### Point Prediction Accuracy
- **RMSE**: The model has an RMSE of roughly 12.5 for point predictions. This relatively high value, compared to the average error of 3.5 points, suggests that while most predictions are quite accurate, there are occasional outliers with significant errors. This insight directs us towards investigating these outliers and refining the model to handle them better.

### Comparison with Professional Bookmakers
- When comparing our model's win/loss prediction accuracy with professional bookmakers like FanDuel, we see our model achieves a 62% accuracy compared to their 65%. This benchmark serves as a valuable target for further refining our predictive capabilities, aiming to close this gap through enhanced data inputs and modeling techniques.

## Visualizations
Below I have included a few different snapshots of visualizations that help explain the model's predictions/performance a bit better.
  1) The first is a  winning/losing streak visualization of the model. This helps identify trends in the model and for which teams it performs well.
  2) The second is a bubble chart which provides a visual summary of how each team's win percentage correlates with the predicted points by the model and shows the actual scoring performance (via bubble size).

### Important Point 1
![Screenshot 2024-01-23 at 3 59 11 PM](https://github.com/Stephe262/NBAScorePredictorML/assets/63209384/9aa54746-dacb-4d72-aeed-f6f074516cdd)

### Important Point 2
![Screenshot 2024-01-30 at 12 51 33 PM](https://github.com/Stephe262/NBAScorePredictorML/assets/63209384/9d2a5432-5623-4481-9b27-4ce3c7c92448)


## Conclusion
- The model shows promise in predicting game outcomes and points scored in NBA games, with a clear path outlined for future enhancements. Refining the model to reduce outlier errors and incorporating more dynamic, real-time data could bridge the gap to the accuracy levels seen in professional betting platforms.
