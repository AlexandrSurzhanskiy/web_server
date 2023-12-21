import io
import pandas as pd

from flask_wtf.file import FileAllowed
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from flask import Flask, request, url_for
from flask import render_template, redirect, send_file

from wtforms.validators import DataRequired
from wtforms import StringField, SubmitField, FileField, SelectField

from ensembles import *

app = Flask(__name__, template_folder='html')
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
app.config['SECRET_KEY'] = 'hello'
Bootstrap(app)


class InitForm(FlaskForm):
    model_list = ['Random Forest MSE', 'Gradient Boosting MSE']
    model = SelectField('Тип модели', choices=model_list)
    n_estimators = StringField('Количество базовых моделей', validators=[DataRequired()], default='100')
    feature_subsample_size = StringField('Размерность подвыборки признаков', validators=[DataRequired()], default='0.9')
    max_depth = StringField('Максимальная глубина дерева', validators=[DataRequired()], default='5')
    learning_rate = StringField('Темп обучения (только для Gradient Boosting MSE!)', validators=[DataRequired()], default='0.01')
    submit = SubmitField('Создать модель')

class TrainDataForm(FlaskForm):
    file_path = FileField('Загрузите обучающую выборку',
                           validators=[
                                        DataRequired('Specify file'),
                                        FileAllowed(['csv'], 'Только CSV!')
    ])
    submit = SubmitField('Загрузить')

class TrainForm(FlaskForm):
    submit = SubmitField('Обучить')

class ValidateForm(FlaskForm):
    submit = SubmitField('Сделать предсказание')

class ValidateDataForm(FlaskForm):
    file_path = FileField('Загрузите валидационную выборку',
                           validators=[
                                        DataRequired('Specify file'),
                                        FileAllowed(['csv'], 'Только CSV!')
    ])
    submit = SubmitField('Загрузить')

class Model:
    model = None
    type = None
    n_estimators = None
    feature_subsample_size = None
    max_depth = None
    learning_rate = None

class TrainData:
    file = None
    X = None
    y = None
    train_rmse = None

class ValidateData:
    X = None
    y = None

model = Model()
train_data = TrainData()
validate_data = ValidateData()


@app.route('/', methods=['POST', 'GET'])
def init_model():
    init_form = InitForm()

    if request.method == 'POST' and init_form.validate_on_submit():
        try:
            n_estimators = int(init_form.n_estimators.data)
            model.n_estimators = n_estimators
            feature_subsample_size = float(init_form.feature_subsample_size.data)
            model.feature_subsample_size = feature_subsample_size
            max_depth = int(init_form.max_depth.data)
            model.max_depth = max_depth
            learning_rate = float(init_form.learning_rate.data)
            model.learning_rate = learning_rate
            if init_form.model == 'Random Forest MSE':
                model.type = 'Random Forest MSE'
                model.model = RandomForestMSE(n_estimators=n_estimators,
                                              feature_subsample_size=feature_subsample_size,
                                              max_depth=max_depth,
                                             )
            else:
                model.type = 'Gradient Boosting MSE'
                model.model = GradientBoostingMSE(n_estimators=n_estimators,
                                                  feature_subsample_size=feature_subsample_size,
                                                  max_depth=max_depth,
                                                  learning_rate=learning_rate
                                                 )
            return redirect(url_for('load_train_data'))
        except Exception as exc:
            app.logger.info(f'Exception: {exc}')
    return render_template('init.html', init_form=init_form)

@app.route('/load_train_data', methods=['POST', 'GET'])
def load_train_data():
    train_data_form = TrainDataForm()

    if request.method == 'POST' and train_data_form.validate_on_submit():
        try:
            stream = io.StringIO(train_data_form.file_path.data.stream.read().decode("UTF8"), newline=None)
            df = pd.read_csv(stream)
            train_data.file = df
            df['date'] = pd.to_datetime(df['date'])
            df.loc[df['yr_renovated'] == 0, ['yr_renovated']] = df['yr_built']
            df['yr_renovated'] = df['date'].dt.year - df['yr_renovated']
            df['date'] = df['date'].dt.day_of_year
            df = df.drop(['id', 'sqft_above', 'yr_built'], axis=1)

            train_data.X = df.drop(['price'], axis=1)
            train_data.y = df['price']
            train_data.X = train_data.X.to_numpy()
            train_data.y = train_data.y.to_numpy()
            
            return redirect(url_for('train'))
        except Exception as exc:
            app.logger.info(f'Exception: {exc}')
    return render_template('load_train_data.html', train_data_form=train_data_form)

@app.route('/train', methods=['POST', 'GET'])
def train():
    train_form = TrainForm()
    model_output = 'Неопределено'
    next_page = ''
    if request.method == 'POST' and train_form.validate_on_submit():
        try:
            model.model.fit(train_data.X, train_data.y)
            y_pred_train = model.model.predict(train_data.X)
            model_output = mean_squared_error(train_data.y, y_pred_train, squared=False)
            train_data.train_rmse = model_output
            next_page = 'Перейти к загрузке данных для валидации'
        except Exception as exc:
            app.logger.info(f'Exception: {exc}')
    return render_template('train.html', train_form=train_form, model_output=model_output, next_page=next_page)

@app.route('/load_validate_data', methods=['POST', 'GET'])
def load_validate_data():
    validate_data_form = ValidateDataForm()
    model_info = 'Получить информацию о модели'
    if request.method == 'POST' and validate_data_form.validate_on_submit():
        try:
            stream = io.StringIO(validate_data_form.file_path.data.stream.read().decode("UTF8"), newline=None)
            df = pd.read_csv(stream)
            df['date'] = pd.to_datetime(df['date'])
            df.loc[df['yr_renovated'] == 0, ['yr_renovated']] = df['yr_built']
            df['yr_renovated'] = df['date'].dt.year - df['yr_renovated']
            df['date'] = df['date'].dt.day_of_year
            df = df.drop(['id', 'sqft_above', 'yr_built'], axis=1)
            
            if 'price' in df.columns:
                validate_data.y = df['price'].to_numpy()
                df = df.drop(['price'], axis=1)

            validate_data.X = df.to_numpy()
            
            return redirect(url_for('validate'))
        except Exception as exc:
            app.logger.info(f'Exception: {exc}')
    return render_template('load_validate_data.html', validate_data_form=validate_data_form, model_info=model_info)

@app.route('/validate', methods=['POST', 'GET'])
def validate():
    validate_form = ValidateForm()
    model_info = 'Получить информацию о модели'
    get_pred = ''
    model_output = 'Неопределено'
    if validate_form.validate_on_submit():
        try:
            y_pred = model.model.predict(validate_data.X)
            if validate_data.y is not None:
                model_output = mean_squared_error(validate_data.y, y_pred, squared=False)
            prediction = pd.DataFrame({'price': y_pred})
            prediction.to_csv('prediction.csv', index=False)
            get_pred = 'Получить предсказание'
        except Exception as exc:
            app.logger.info(f'Exception: {exc}')
    return render_template('validate.html', validate_form=validate_form, model_info=model_info, get_pred=get_pred, model_output=model_output)

@app.route('/prediction')
def prediction():
    return send_file('prediction.csv', as_attachment=True)

@app.route('/info')
def info():
    download_train_data = ''
    try:
        train_data.file.to_csv('train_data.csv', index=False)
        download_train_data = 'Получить обучающую выборку' 
    except Exception as exc:
            app.logger.info(f'Exception: {exc}')   
    return render_template('info.html', 
                           type = model.type,
                           n_estimators=model.n_estimators,
                           feature_subsample_size=model.feature_subsample_size,
                           max_depth=model.max_depth,
                           learning_rate = model.learning_rate,
                           download_train_data=download_train_data,
                           train_rmse=train_data.train_rmse
                          )

@app.route('/train_dataset')
def train_dataset():
    return send_file('train_data.csv', as_attachment=True)