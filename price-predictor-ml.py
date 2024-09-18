!pip install scikit-learn
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
!pip install seaborn
import seaborn as sns
import glob
!pip install category_encoders
from category_encoders import OneHotEncoder
!pip install ipywidgets
from ipywidgets import Dropdown, FloatSlider, IntSlider, interact,Output


from category_encoders import OneHotEncoder
#defining a wrangle function to import and clean all my data
def wrangle(filepath):
    # Read CSV file
    df = pd.read_csv(filepath, encoding='ISO-8859-1')

    # Subset data: Apartments in "Capital Federal", less than 400,000
    mask_ba = df["place_with_parent_names"].str.contains("Capital Federal")
    mask_apt = df["property_type"] == "apartment"
    mask_price = df["price_aprox_usd"] < 400_000
    df = df[mask_ba & mask_apt & mask_price]

    # Subset data: Remove outliers for "surface_covered_in_m2"
    low, high = df["surface_covered_in_m2"].quantile([0.1, 0.9])
    mask_area = df["surface_covered_in_m2"].between(low, high)
    df = df[mask_area]

    # Split "lat-lon" column
    df[["lat", "lon"]] = df["lat-lon"].str.split(",", expand=True).astype(float)
    df.drop(columns="lat-lon", inplace=True)

    # Get place name
    df["neighborhood"] = df["place_with_parent_names"].str.split("|", expand=True)[3]
    df.drop(columns="place_with_parent_names", inplace=True)
    
    #drop features with high null count
    df.drop(columns=["floor","expenses"], inplace = True)
    
    #drop low and high cardinality categorical variables
    df.drop(columns=["operation", "property_type", "currency", "properati_url"], inplace=True)
    
    # drop leaky columns
    df.drop(columns=[
        'price',
        'price_aprox_local_currency',
        'price_per_m2',
        'price_usd_per_m2'], 
           inplace=True)
    
    #drop columns with multicollinearity
    df.drop(columns=["surface_total_in_m2", "rooms"], inplace =True)
    return df
    
#using glob to create a list that contains the filenames for all the Buenos Aires real estate CSV files 
files = glob.glob("C:/Users/User/Desktop/buenos-aires-analysis/buenos-aires-real-estate-data-*.csv")

#Use your wrangle function in a list comprehension to create a list containing the cleaned DataFrames for the filenames
#I collected in files
frames = [wrangle(file) for file in files]
frames[0].head()

#concatenate items in frames into a single DataFrame df
df = pd.concat(frames, ignore_index=True)

#EDA
#Plot a correlation heatmap of the remaining numerical features in df. Since "price_aprox_usd" will be the target,there's need to
#include it in the heatmap.
corr = df.select_dtypes("number").drop(columns="price_aprox_usd").corr()
corr
sns.heatmap(corr)
plt.show()

#Splitting Data
target = "price_aprox_usd"
y_train= df[target]
features = ["surface_covered_in_m2", "lat", "lon", "neighborhood"]
X_train = df[features]


#MODEL BUILDING
#Calculating the baseline mean absolute error for the model.
y_mean= y_train.mean()
y_pred_baseline = [y_mean] * len(y_train)
mean_absolute_error(y_train, y_pred_baseline)
print("Mean apt price:", round(y_mean, 2))
print("Baseline MAE:", mean_absolute_error(y_train, y_pred_baseline))

#Iteration
#Creating a pipeline named model that contains a OneHotEncoder, SimpleImputer, and Ridge predictor.
model = make_pipeline(
  OneHotEncoder(use_cat_names=True),
  SimpleImputer(),
  Ridge()
)
model.fit(X_train, y_train)

#Model Evaluation
y_pred_training = model.predict(X_train)
mean_absolute_error(y_train, y_pred_training)
#importing test data
X_test = pd.read_csv("C:/Users/User/Desktop/buenos-aires-analysis/buenos-aires-test-data.csv", encoding='ISO-8859-1')
#dropping the index column
X_test = X_test.drop(columns=['Unnamed: 0'])

X_test.info()

#predicting
y_pred_test = pd.Series(model.predict(X_test))
y_pred_test.head()

#MODEL DEPLOYMENT
#wrapping the model in a function so that one can provide inputs and then receive a prediction as output.
#Creating a function make_prediction function
def make_prediction(area, lat, lon, neighborhood):
    
    data = {
    "surface_covered_in_m2": area,
    "lat": lat,
    "lon": lon,
    "neighborhood":neighborhood
    }
    df = pd.DataFrame(data, index=[0])
    prediction= model.predict(df).round(2)[0]
    return f"Predicted apartment price: ${prediction}"
  
#testing the function
make_prediction(110, -34.60, -58.46, "Villa Crespo")
#it works!

#creating an interactive dashboard  
from IPython.display import display


interact(
    make_prediction,
    area=IntSlider(
        min=X_train["surface_covered_in_m2"].min(),
        max=X_train["surface_covered_in_m2"].max(),
        value=X_train["surface_covered_in_m2"].mean(),
    ),
    lat=FloatSlider(
        min=X_train["lat"].min(),
        max=X_train["lat"].max(),
        step=0.01,
        value=X_train["lat"].mean(),
    ),
    lon=FloatSlider(
        min=X_train["lon"].min(),
        max=X_train["lon"].max(),
        step=0.01,
        value=X_train["lon"].mean(),
    ),
    neighborhood=Dropdown(options=sorted(X_train["neighborhood"].unique())),
);
