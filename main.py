import pandas as pd
import numpy as np
import io
import random
from flask import (
    Flask,
    request,
    render_template,
    session,
    redirect,
    render_template_string,
    Response,
)
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

cars = pd.read_csv("Cheapestelectriccars-EVDatabase.csv")
cars_view = pd.read_csv("Cheapestelectriccars-EVDatabase.csv")
cars_view.drop(columns=["PriceinGermany", "PriceinUK"], inplace=True)
cars.columns = [
    "name",
    "subtitle",
    "acceleration_in_sec",
    "top_speed_km_per_h",
    "range_km",
    "efficiency_wh_per_hour",
    "fast_charge_speed_km_per_h",
    "drive",
    "number_of_seats",
    "price_in_euros",
    "price_to_drop",
]
cars.drop(columns="price_to_drop", inplace=True)
numeric_columns = cars.loc[
    :,
    [
        "acceleration_in_sec",
        "top_speed_km_per_h",
        "range_km",
        "efficiency_wh_per_hour",
        "fast_charge_speed_km_per_h",
        "number_of_seats",
        "price_in_euros",
    ],
]

cars.acceleration_in_sec = cars.acceleration_in_sec.str.replace("sec", "").astype(
    "float"
)
cars.top_speed_km_per_h.str.replace("km/h", "").astype("int")
cars.range_km = cars.range_km.str.replace("km", "").astype("int")
cars.efficiency_wh_per_hour = cars.efficiency_wh_per_hour.str.replace(
    "Wh/km", ""
).astype("int")
cars.fast_charge_speed_km_per_h = cars.fast_charge_speed_km_per_h.str.replace(
    "km/h", ""
)
cars.price_in_euros = (
    cars.price_in_euros.str.replace("€", "").str.replace(",", ".").astype("float")
)
cars.loc[cars.price_in_euros.isnull(), "price_in_euros"] = round(
    cars.price_in_euros.mean(), 2
)
info = cars.describe()

app = Flask(__name__)


@app.route("/", methods=("POST", "GET"))
def html_table():
    return render_template(
        "index.html",
        tables=[cars_view.to_html(classes="data")],
        titles=cars_view.columns.values,
    )


@app.route("/view", methods=("POST", "GET"))
def html_table1():
    return render_template(
        "view.html",
        tables=[info.to_html(classes="data")],
        titles=cars_view.columns.values,
    )
