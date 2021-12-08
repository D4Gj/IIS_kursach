import json
import pickle

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
    url_for,
    make_response,
    current_app,
    send_file,
)
from sklearn.tree import DecisionTreeClassifier
from fpdf import FPDF
import config
import joblib

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

cars = pd.read_csv("Cheapestelectriccars-EVDatabase.csv")
cars_view = pd.read_csv("Cheapestelectriccars-EVDatabase.csv")
cars_view["PriceinGermany"] = cars_view["PriceinGermany"].replace(np.NAN, "no info")
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
    cross_validation_start()
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
    create_pdf("Отчёт по самой минимальной цене", f"Самой дешёвой машиной является {name} с ценой в {price} евро",
               "static/min.png",
               "min_price")
    return render_template(
        "min_price.html",
        name=name,
        price=price
    )


@app.route("/min_price.pdf")
def send_pdf_min():
    return send_file(config.STATIC_FOLDER + "\min_price.pdf")


@app.route("/max_speed", methods=("POST", "GET"))
def html_max_speed():
    print(cars.tail())
    name, speed = max_speed_find()
    create_pdf("Отчёт по максимальной скорости", f"Самой быстрой машиной является {name}, со скорость {speed}.",
               image_path="static/max_speed.png", save_name="max_speed")
    return render_template(
        "max_speed.html",
        name=name,
        speed=speed
    )


@app.route("/max_speed.pdf")
def send_pdf_max():
    return send_file(config.STATIC_FOLDER + "\max_speed.pdf")


@app.route("/mean_range", methods=("POST", "GET"))
def html_mean_range():
    print(cars.tail())
    speed = mean_range_find()
    create_pdf(title="Средняя дистанция прохождения",
               text=f"В среднем электрокары проезжают {speed} километров.",
               image_path="static/mean_range.png",
               save_name="mean_range")
    return render_template(
        "mean_range.html",
        speed=speed,
    )


@app.route("/mean_range.pdf")
def send_pdf_mean_range():
    return send_file(config.STATIC_FOLDER + "\mean_range.pdf")


@app.route("/classification", methods=("POST", "GET"))
def html_clustering():
    print(cars.tail())
    names, values, score = classification()
    print(names)
    print(list(values))
    clusters = pd.DataFrame(data=[values], columns=names, index=["Важность"])
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font('DejaVu', '', 'font/DejaVuSansCondensed.ttf', uni=True)
    pdf.set_font('DejaVu', '', 14)
    pdf.write(txt="Классификация", h=15)
    pdf.ln(25)
    pdf.write(txt=clusters.to_string(col_space=21), h=7)
    pdf.ln(10)
    pdf.write(
        txt="Исходя из классификации, можно сказать, что наибольшее влияние на эффективность работы электродвигателя "
            "оказывает дальность расстояния, а именно, сколько километров сможет проехать электроавтомобиль на одном "
            "заряде. Качество модели:" + str(
            score), h=7)
    pdf.output("static/classification.pdf")
    return render_template(
        "classification.html",
        score=round(score, 5),
        tables=[clusters.to_html(classes="data")]
    )


@app.route("/classification.pdf")
def send_pdf_classification():
    return send_file(config.STATIC_FOLDER + "\classification.pdf")


@app.route("/cross_validation", methods=("POST", "GET"))
def cross_validation():  # Кросс-валидация качество модели
    return render_template("cross_validation.html", validation=str(cross_validation_start()))


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
    names = ["Макс скорость", "Макс дистанция", "Количество сидений"]
    save_file(clf, "static/model.pkl")
    return names, clf.feature_importances_, clf.score(x, y),


def create_pdf(title, text, image_path, save_name):
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font('DejaVu', '', 'font/DejaVuSansCondensed.ttf', uni=True)
    pdf.set_font('DejaVu', '', 14)
    pdf.write(txt=title, h=15)
    pdf.ln(20)
    pdf.image(image_path, x=10, y=20, w=170)
    pdf.ln(130)  # ниже на 85
    pdf.write(
        txt=text,
        h=7)
    pdf.output("static/" + save_name + ".pdf")


def cross_validation_start():
    x = cars[['top_speed_km_per_h', 'range_km', "number_of_seats"]]
    y = cars['efficiency_wh_per_hour']

    clf = load_file("static/model.pkl")

    cv_results = cross_val_score(clf, x, y, cv=3)

    return cv_results.mean()


def load_file(filename):
    return joblib.load(filename)


def save_file(model, filename):
    joblib.dump(model, filename)
