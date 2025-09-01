# pris_labs
Веб-приложение для анализа данных и машинного обучения с использованием Flask. Приложение позволяет анализировать данные из CSV файлов, применять машинное обучение и визуализировать данные.
Пользоватеелю требуется всего лишь загрузить данные в разделе "Выбрать файл" и нажать кнопку "Загрузить и проанализировать", далее приложение сделает все само! Для обучения модели пользователь должен выбрать целевую переменную из выпадающего списка.

Основные моменты:
1. Инициализация Flask и папок
```
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)

UPLOAD_FOLDER = 'uploads'
PLOT_FOLDER = 'static/plots'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)
```
Настройка путей к файлам и создание необходимых папок
2. Анализ данных
```
def analyze_data(df):
    # Heatmap корреляции
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.savefig(os.path.join(PLOT_FOLDER, 'correlation.png'))
    
    # Гистограммы распределения
    numeric_df.hist(bins=15, color='#ff4b2b', alpha=0.7)
    plt.savefig(os.path.join(PLOT_FOLDER, 'numeric_distribution.png'))
```
Создание графиков анализа данных
3. Обучение модели ML
```
X = df.drop(columns=[target_column])
y = df[target_column]
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
```
Подготовка данных и обучение модели (Random Forest)
