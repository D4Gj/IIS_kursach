import pandas as pd
import numpy as np
import io
import random
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering

from flask import (
    Flask,
    request,
    render_template,
    session,
    redirect,
    render_template_string,
    Response,
)
from sklearn.tree import DecisionTreeClassifier

import config

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

cars = pd.read_csv("Cheapestelectriccars-EVDatabase.csv")
cars_view = pd.read_csv("Cheapestelectriccars-EVDatabase.csv")
cars_view.loc[cars.price_in_euros.isnull(), "price_in_euros"] = round(
    cars.price_in_euros.mean(), 2
)
cars_view.drop(columns=["PriceinUK"], inplace=True)
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
cars.top_speed_km_per_h = cars.top_speed_km_per_h.str.replace("km/h", "").astype("int")
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
info = cars.describe(percentiles=[])

app = Flask(
    __name__, static_folder=config.STATIC_FOLDER, template_folder=config.TEMPLATE_FOLDER
)


@app.route("/", methods=("POST", "GET"))
def html_table():
    return render_template(
        "index.html",
        tables=[cars_view.to_html(classes="data")],
        titles=cars_view.columns.values,
    )


@app.route("/view", methods=("POST", "GET"))
def html_table1():
    print(cars.tail())
    return render_template(
        "view.html",
        tables=[info.to_html(classes="data")],
    )


@app.route("/min_price", methods=("POST", "GET"))
def html_min():
    print(cars.tail())
    name, price = min_price_find()
    price *= 1000
    return render_template(
        "min_price.html",
        name=name,
        price=price
    )


@app.route("/max_speed", methods=("POST", "GET"))
def html_max_speed():
    print(cars.tail())
    name, speed = max_speed_find()
    return render_template(
        "max_speed.html",
        name=name,
        speed=speed
    )


@app.route("/mean_range", methods=("POST", "GET"))
def html_mean_range():
    print(cars.tail())
    speed = mean_range_find()
    return render_template(
        "mean_range.html",
        speed=speed,
    )


@app.route("/classification", methods=("POST", "GET"))
def html_clustering():
    print(cars.tail())
    names, values, score = classification()
    print(names)
    print(list(values))
    clusters = pd.DataFrame(data=[values], columns=names, index=["Важность"])

    return render_template(
        "classification.html",
        score=round(score, 5),
        tables=[clusters.to_html(classes="data")]
    )


# --Поиск машины с минимальной скоростью--
def min_price_find():
    min_price = cars['price_in_euros'][0]
    min_index = 0
    for i in range(len(cars['price_in_euros'])):
        if cars['price_in_euros'][i] < min_price:
            min_price = cars['price_in_euros'][i]
            min_index = i

    fig, ax = plt.subplots()
    ax.set_title('Распределение цен автомобилей')
    plt.style.context('bmh')
    ax.set_xlabel('Цена')
    ax.set_ylabel('Количество машин')
    plt.hist(cars['price_in_euros'], bins=15)
    plt.grid('on')
    plt.savefig('static/min.png')
    return cars['name'][min_index], min_price


# --Поиск машины с максимальной скоростью--
def max_speed_find():
    max_speed = cars['top_speed_km_per_h'][0]
    max_index = 0
    for i in range(len(cars['top_speed_km_per_h'])):
        if cars['top_speed_km_per_h'][i] > max_speed:
            max_speed = cars['top_speed_km_per_h'][i]
            max_index = i

    fig, ax = plt.subplots()
    ax.set_title('Распределение скоростей автомобилей')
    plt.style.context('bmh')
    ax.set_xlabel('Скорость')
    ax.set_ylabel('Количество машин')
    plt.hist(cars['top_speed_km_per_h'], bins=15)
    plt.grid('on')
    plt.savefig('static/max_speed.png')
    return cars['name'][max_index], max_speed


# --Поиск средней дальности хода автомобилей--
def mean_range_find():
    summ = 0
    for i in range(len(cars['range_km'])):
        summ = summ + cars['range_km'][i]

    result = summ / len(cars['range_km'])

    fig, ax = plt.subplots()
    ax.set_title('Дальность хода автомобилей')
    plt.style.context('bmh')
    ax.set_xlabel('Дальность')
    ax.set_ylabel('Количество машин')
    plt.hist(cars['range_km'], bins=15)
    plt.grid('on')
    plt.savefig('static/mean_range.png')
    return round(result, 2)


def classification():
    x = cars[['top_speed_km_per_h', 'range_km', "number_of_seats"]]
    y = cars['efficiency_wh_per_hour']
    clf = DecisionTreeClassifier()
    clf.fit(x, y)
    print(clf.feature_names_in_)
    # Вывод важности
    print(clf.feature_importances_)
    print(clf.score(x, y))  # верность классификации
    names = ["Максимальная скорость", "Максимальная дистация", "Количество сидений"]
    return names, clf.feature_importances_, clf.score(x, y),
