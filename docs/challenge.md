# Explanations and Docs

In this MD I'll explain my choices and paths that I've went through to
solve this challenge.

It's important to cite in this part of the docs that the endpoint of the
deployed application is: https://challenge-latam-914044025861.us-central1.run.app

## Explanation on model selection

After analyzing the classification results, XGBoost with
Feature Engineering (FE) and Balance and Logistic Regression
with FE and Balance performed similarly, with XGBoost showing a slight
improvement of 1% in the F1-score.

However, when evaluating the training and prediction times, Logistic
Regression emerged as the clear winner in terms of computational
efficiency. Specifically, XGBoost took a mean of 859.1 ms to fit the
model and 10.49 ms for prediction (with 1 run per loop), whereas
Logistic Regression took only 68.93 ms to fit the model and 1.27 ms
for prediction (with 100 and 1000 runs per loop, respectively).
This shows that Logistic Regression is over 10 times faster during
the fit step and about 8 times faster during prediction.

While the 1% improvement in F1-score from XGBoost might seem small, it is
an important metric when dealing with imbalanced classes.
The F1-score balances precision (the proportion of correctly predicted
positive instances) and recall (the ability to identify all positive instances).
In problems with imbalanced classes, where one class may be
underrepresented, the F1-score ensures that the model does not bias
toward the majority class and misses critical instances of the
minority class. This balance is crucial, especially when missing
instances of the minority class could have significant consequences, such
as in fraud detection or medical diagnosis.

However, given the substantial difference in computation time—with
Logistic Regression being over 10 times faster in fitting the model
and about 8 times faster in prediction—the choice of Logistic Regression
becomes optimal in this case, particularly in real-time or high-throughput
environments where speed is critical. The 1% improvement in F1-score from
XGBoost does not justify its higher computational cost in this case.

If further testing indicates that the slight improvement in performance
from XGBoost is critical for the application, future optimizations or
parallel processing could be considered to reduce the time impact.

## Model

To complete the model.py implementation, I ensured to strictly maintain the
original class structure, as required. The structure already had a clear and
modular approach with methods for fit, predict, and preprocess, which I found
effective for organizing the code. Instead of introducing new classes or changing
the existing setup, I focused on enhancing the methods to meet the project’s needs
while preserving the original design.

The preprocess method was adapted to handle both feature extraction and the optional
creation of a target column. This modification allowed the method to remain flexible,
suitable for both training (where the target column is needed) and prediction (where it isn't).
This keeps the code simple and reusable for different tasks.

I also introduced the create_target_column method to isolate the logic for generating
the target column (delay). This separation ensures that any changes to how the delay is
calculated won’t affect the overall data processing pipeline, which maintains the clarity
of the original structure.

In the fit method, I added class weights to the logistic regression model to address class
imbalance. This ensures that the model accurately predicts delays, especially for minority classes,
by assigning appropriate weights to each class.

The predict method was designed to load the pre-trained model from a serialized file
(flight_delay_logreg_model.joblib), which avoids retraining the model every time a prediction is made.
This design choice streamlines the process and ensures consistent predictions.

Lastly, I kept the use of joblib for model serialization, as it efficiently handles larger models,
such as the logistic regression model, and allows for easy storage and reuse.

By maintaining the original structure, I ensured that the class remained clear, modular, and easy
to extend, while also enhancing the functionality to meet the project’s requirements.

## API

Starting with the /health endpoint, I kept it in the same way that were when I've started.

For the /predict endpoint, I had to extend the code to handle input validation and integrate the
prediction logic. The original code defined the endpoint, but additional logic was required to handle
requests properly. I began by verifying that the request included a flights key. If this key was missing,
an HTTP 400 Bad Request error was raised.

I then moved on to checking the integrity of the input data. I validated that the provided data was not
empty and ensured that the required columns (OPERA, TIPOVUELO, and MES) were present in the request.
Any missing columns resulted in a clear error message, which improved the user experience and helped
ensure correct input formatting.

For each row in the dataset, I verified the values of MES, TIPOVUELO, and OPERA. This validation ensured
that the MES value was an integer between 1 and 12, TIPOVUELO was a valid string, and OPERA was a non-empty string.
These checks prevented any invalid data from being passed to the model, which could have resulted in errors during prediction.

Finally, I added error handling to catch any unforeseen issues during the prediction process. If an exception occurred,
the API returned a 500 Internal Server Error with a helpful message detailing the issue.
