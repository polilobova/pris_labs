#http://localhost:5501
import os
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, flash
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import io

# Инициализация Flask приложения
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)
app.secret_key = 'your-secret-key-here'

# Создание папок для загрузок и графиков
UPLOAD_FOLDER = 'uploads'
PLOT_FOLDER = 'static/plots'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)

global_df = None
global_model = None

def analyze_data(df):
    """Анализ данных и создание визуализаций"""
    plots = {}
    # очищаем папку с графиками перед созданием новых
    for file in os.listdir(PLOT_FOLDER):
        if file.endswith('.png'):
            os.remove(os.path.join(PLOT_FOLDER, file))
    
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()

    stats = df.describe().to_html(classes='table table-striped')
    
    # Heatmap correlation
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 1:
        plt.figure(figsize=(10, 8))
        corr = numeric_df.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Матрица корреляции')
        plt.tight_layout()
        correlation_path = 'correlation.png'
        full_path = os.path.join(PLOT_FOLDER, correlation_path)
        plt.savefig(full_path, dpi=100, bbox_inches='tight')
        plt.close()
        plots['correlation'] = correlation_path
    
    # распределение числовых признаков
    if len(numeric_df.columns) > 0:
        fig, axes = plt.subplots(nrows=(len(numeric_df.columns) + 1) // 2, ncols=2, figsize=(12, 8))
        if len(numeric_df.columns) == 1:
            axes = [axes]
        
        axes = np.array(axes).flatten()
        
        for i, col in enumerate(numeric_df.columns):
            if i < len(axes):
                numeric_df[col].hist(bins=15, ax=axes[i], color='#ff4b2b', alpha=0.7, edgecolor='black')
                axes[i].set_title(col, fontsize=10, fontweight='bold')
                axes[i].tick_params(axis='both', which='major', labelsize=8)
        
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.suptitle('Распределение числовых признаков', fontsize=12, fontweight='bold')
        plt.tight_layout()
        numeric_dist_path = 'numeric_distribution.png'
        full_path = os.path.join(PLOT_FOLDER, numeric_dist_path)
        plt.savefig(full_path, dpi=100, bbox_inches='tight')
        plt.close()
        plots['numeric_distribution'] = numeric_dist_path
    
    return info_str, stats, plots

def train_model(df, target_column):
    """Обучение модели машинного обучения"""
    try:
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        le = LabelEncoder()
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            X[col] = le.fit_transform(X[col].astype(str))
        
        if y.dtype == 'object':
            y = le.fit_transform(y)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # обучение
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # матрица ошибок
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Матрица ошибок', fontweight='bold')
        plt.ylabel('Истинные значения')
        plt.xlabel('Предсказанные значения')
        plt.tight_layout()
        cm_path = 'confusion_matrix.png'
        full_path = os.path.join(PLOT_FOLDER, cm_path)
        plt.savefig(full_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # отчет классификации
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        
        return model, accuracy, cm_path, report_df.to_html(classes='table table-striped')
    
    except Exception as e:
        raise Exception(f"Ошибка при обучении модели: {str(e)}")

@app.route('/', methods=['GET', 'POST'])
def index():
    global global_df, global_model
    
    data_info = None
    data_stats = None
    plots = {}
    accuracy = None
    classification_report_html = None
    target_column = None
    columns_list = []
    
    try:
        if request.method == 'POST':
            if 'file' in request.files:
                file = request.files['file']
                
                if file and file.filename != '':
                    
                    if not (file.filename.endswith('.csv') or file.filename.endswith(('.xls', '.xlsx'))):
                        flash('Неподдерживаемый формат файла. Используйте CSV или Excel.', 'error')
                        return render_template('index.html')
                    
                    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
                    
                    try:
                        file.save(file_path)
                        
                        try:
                            if file.filename.endswith('.csv'):
                                global_df = pd.read_csv(file_path)
                            else:
                                global_df = pd.read_excel(file_path)
                            
                            if global_df.empty:
                                flash('Файл пустой или не содержит данных', 'error')
                                return render_template('index.html')
                            
                            # анализ данных
                            data_info, data_stats, plots = analyze_data(global_df)
                            
                            columns_list = global_df.columns.tolist()
                            
                            flash('Файл успешно загружен и проанализирован', 'success')
                            
                        except Exception as e:
                            error_msg = f'Ошибка при чтении файла: {str(e)}'
                            flash(error_msg, 'error')
                            return render_template('index.html')
                        
                    except Exception as e:
                        error_msg = f'Ошибка при сохранении файла: {str(e)}'
                        flash(error_msg, 'error')
                        return render_template('index.html')
            
            elif 'train_model' in request.form and global_df is not None:
                target_column = request.form.get('target_column')
                
                if not target_column:
                    flash('Выберите целевую переменную', 'error')
                    return render_template('index.html')
                
                try:
                    global_model, accuracy, cm_path, classification_report_html = train_model(
                        global_df, target_column
                    )
                    plots['confusion_matrix'] = cm_path
                    flash('Модель успешно обучена', 'success')
                    
                    # сохранение модели
                    model_path = os.path.join(UPLOAD_FOLDER, 'trained_model.joblib')
                    joblib.dump(global_model, model_path)
                    
                except Exception as e:
                    error_msg = f'Ошибка при обучении модели: {str(e)}'
                    flash(error_msg, 'error')
    
    except Exception as e:
        error_msg = f'Произошла непредвиденная ошибка: {str(e)}'
        flash(error_msg, 'error')
    
    return render_template('index.html',
                         data_info=data_info,
                         data_stats=data_stats,
                         plots=plots,
                         accuracy=accuracy,
                         classification_report=classification_report_html,
                         columns=columns_list,
                         has_data=global_df is not None)

@app.route('/debug')
def debug():
    import os
    info = {
        'current_dir': os.getcwd(),
        'files': os.listdir('.'),
        'templates_exists': os.path.exists('templates'),
        'static_exists': os.path.exists('static'),
        'templates_content': os.listdir('templates') if os.path.exists('templates') else []
    }
    return info

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5501)