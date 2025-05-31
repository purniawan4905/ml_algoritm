from flask import Flask, render_template, request, redirect, url_for, send_from_directory, send_file
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import uuid
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, mean_squared_error
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.utils.multiclass import unique_labels
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DATA_PATH = os.path.join(UPLOAD_FOLDER, "latest_data.pkl")

# Lokasi folder dataset
DATASET_FOLDER = os.path.join(app.root_path, 'static', 'datasets')

def handle_non_numerical(X):
    X_encoded = X.copy()
    for col in X.columns:
        if X[col].dtype == 'object':
            X_encoded[col] = LabelEncoder().fit_transform(X[col].astype(str))
    return X_encoded


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    chart_filename = None
    importance_filename = None
    dist_chart_data = None
    dist_chart_labels = None
    strategy = None
    algorithm = None
    imputasi_info = ""

    if request.method == "POST":
        file = request.files.get("file")
        strategy = request.form.get("strategy")
        algorithm = request.form.get("algorithm")

        if file and file.filename:
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.filename.endswith('.xlsx'):
                df = pd.read_excel(file)
            else:
                result = "<div class='alert alert-danger'><strong>File tidak valid. Harap upload file CSV atau Excel.</strong></div>"
                return render_template("index.html", result=result)

            df.to_pickle(DATA_PATH)  # Simpan data asli untuk dashboard

            # Pisah fitur dan target
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]

            if y.isnull().any():
                y.fillna(y.mode()[0], inplace=True)

            X = handle_non_numerical(X)

            # Cek apakah ada missing value
            missing_value_exists = X.isnull().values.any()

            if missing_value_exists:
                if strategy == "knn":
                    imputer = KNNImputer(n_neighbors=5)
                    X_imputed = imputer.fit_transform(X)
                elif strategy == "modus":
                    X_imputed = X.copy()
                    for column in X.columns:
                        if X[column].isnull().any():
                            mode = X[column].mode()[0]
                            X_imputed[column] = X_imputed[column].fillna(mode)
                    X_imputed = X_imputed.values
                else:
                    imputer = SimpleImputer(strategy=strategy)
                    X_imputed = imputer.fit_transform(X)

                imputasi_info = f"""
                <div class="alert alert-info" role="alert">
                    Imputasi dilakukan menggunakan strategi: <strong>{strategy}</strong>.
                </div>
                """
            else:
                X_imputed = X.values
                imputasi_info = """
                <div class="alert alert-success" role="alert">
                    Tidak ada missing value, proses imputasi dilewati.
                </div>
                """

            # Gabungkan kembali
            df_cleaned = pd.concat([pd.DataFrame(X_imputed, columns=X.columns), y.reset_index(drop=True)], axis=1).dropna()
            X_cleaned = df_cleaned.iloc[:, :-1]
            y_cleaned = df_cleaned.iloc[:, -1]

            # Reset variabel visual
            chart_filename = None
            importance_filename = None
            dist_chart_data = None
            dist_chart_labels = None

            # Unsupervised learning
            if algorithm in ["kmeans", "pca"]:
                plt.figure(figsize=(8, 6))
                if algorithm == "kmeans":
                    model = KMeans(n_clusters=3, random_state=42)
                    y_pred = model.fit_predict(X_cleaned)
                    plt.scatter(X_cleaned.iloc[:, 0], X_cleaned.iloc[:, 1], c=y_pred, cmap='viridis')
                    plt.title("K-Means Clustering")
                else:
                    model = PCA(n_components=2)
                    components = model.fit_transform(X_cleaned)
                    plt.scatter(components[:, 0], components[:, 1])
                    plt.title("PCA Result")

                chart_filename = f"chart_{uuid.uuid4().hex}.png"
                chart_path = os.path.join(app.config['UPLOAD_FOLDER'], chart_filename)
                plt.tight_layout()
                plt.savefig(chart_path)
                plt.close()

                result = f"""
                {imputasi_info}
                <div class='alert alert-success'><strong>{algorithm.upper()} berhasil dijalankan dan divisualisasikan.</strong></div>
                <a href='/dashboard' class='btn btn-info mt-2'>Lihat Dashboard</a>
                """

            else:
                # Supervised learning
                X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y_cleaned, test_size=0.3, random_state=42)

                if algorithm == "naive_bayes":
                    model = GaussianNB()
                elif algorithm == "linear_regression":
                    model = LinearRegression()
                elif algorithm == "logistic_regression":
                    model = LogisticRegression(max_iter=1000)
                elif algorithm == "decision_tree":
                    model = DecisionTreeClassifier()
                elif algorithm == "random_forest":
                    model = RandomForestClassifier()
                elif algorithm == "svm":
                    model = SVC()
                elif algorithm == "ann":
                    model = MLPClassifier(max_iter=1000)
                else:
                    model = GaussianNB()

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                if algorithm == "linear_regression":
                    mse = mean_squared_error(y_test, y_pred)
                    result = f"""
                    {imputasi_info}
                    <p><strong>Mean Squared Error:</strong> {mse:.2f}</p>
                    <a href='/dashboard' class='btn btn-info mt-2'>Lihat Dashboard</a>
                    """
                else:
                    if len(unique_labels(y_test)) > 1:
                        plt.figure(figsize=(6, 6))
                        disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
                        plt.title("Confusion Matrix")
                        chart_filename = f"conf_matrix_{uuid.uuid4().hex}.png"
                        chart_path = os.path.join(app.config['UPLOAD_FOLDER'], chart_filename)
                        plt.tight_layout()
                        plt.savefig(chart_path)
                        plt.close()

                    report = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report).transpose().round(2)
                    report_html = report_df.to_html(classes="table table-bordered")
                    result = f"""
                    {imputasi_info}
                    {report_html}
                    <a href='/dashboard' class='btn btn-info mt-2'>Lihat Dashboard</a>
                    """

                    # Feature importance (tree-based)
                    if algorithm in ["random_forest", "decision_tree"]:
                        importances = model.feature_importances_
                        feat_imp_df = pd.DataFrame({
                            'Feature': X_cleaned.columns,
                            'Importance': importances
                        }).sort_values(by='Importance', ascending=False)

                        plt.figure(figsize=(8, 5))
                        sns.barplot(x='Importance', y='Feature', data=feat_imp_df)
                        plt.title("Feature Importance")
                        plt.tight_layout()
                        importance_filename = f"importance_{uuid.uuid4().hex}.png"
                        importance_path = os.path.join(app.config['UPLOAD_FOLDER'], importance_filename)
                        plt.savefig(importance_path)
                        plt.close()

                    # Distribusi kelas
                    if algorithm in ["naive_bayes", "logistic_regression", "decision_tree", "random_forest", "svm", "ann"]:
                        dist_counts = y_cleaned.value_counts()
                        dist_chart_labels = dist_counts.index.tolist()
                        dist_chart_data = dist_counts.values.tolist()

            # Save untuk dashboard
            df_cleaned.to_pickle(DATA_PATH)
        else:
            result = "<div class='alert alert-warning'><strong>Harap upload file terlebih dahulu.</strong></div>"

    return render_template("index.html",
        result=result,
        chart_filename=chart_filename,
        strategy=strategy,
        algorithm=algorithm,
        importance_filename=importance_filename,
        chart_data=dist_chart_data,
        chart_labels=dist_chart_labels,
    )


def index():
    dataset = "iris"  # default
    if request.method == 'POST':
        dataset = request.form.get('use_sample', 'iris')

    file_path = os.path.join(DATASET_FOLDER, f"{dataset}.csv")
    df = pd.read_csv(file_path)

    table_html = df.to_html(classes="table table-bordered table-hover", index=False)

    return render_template("dataset.html", dataset=dataset, table=table_html)

@app.route('/download/<dataset_name>')
def download_file(dataset_name):
    file_path = os.path.join(DATASET_FOLDER, f"{dataset_name}.csv")
    return send_file(file_path, as_attachment=True)

@app.route("/sample/<dataset>")
def sample(dataset):
    if dataset == "iris":
        from sklearn.datasets import load_iris
        data = load_iris(as_frame=True)
        df = pd.concat([data.data, pd.Series(data.target, name='target')], axis=1)
    elif dataset == "boston":
        url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
        df = pd.read_csv(url)
    elif dataset == "titanic":
        df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
    else:
        return "<p>Dataset tidak ditemukan.</p>"

    table_html = df.head(20).to_html(classes="table table-striped table-bordered", index=False)
    return render_template("sample.html", dataset=dataset.capitalize(), table=table_html)


@app.route("/dashboard")
def dashboard():
    if not os.path.exists(DATA_PATH):
        return "<p><strong>Belum ada data yang diunggah.</strong></p>"

    df = pd.read_pickle(DATA_PATH)

    stats_html = df.describe().transpose().round(2).to_html(classes="table table-bordered")

    # Correlation heatmap
    plt.figure(figsize=(10, 6))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()

    heatmap_file = f"heatmap_{uuid.uuid4().hex}.png"
    heatmap_path = os.path.join(UPLOAD_FOLDER, heatmap_file)
    plt.savefig(heatmap_path)
    plt.close()

    # Get optional feature importance file from the last run, if exists
    # Note: You might want to store this filename in a session or a file for better persistence,
    # here just list last saved file with prefix importance_ as simplest approach

    importance_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.startswith("importance_") and f.endswith(".png")]
    importance_filename = sorted(importance_files, reverse=True)[0] if importance_files else None

    # Similarly for distribution chart (not saved as image, rendered by Chart.js)

    # Optional: load chart data and labels from somewhere persistent (session/db)
    # For now, pass None. You may extend by storing JSON or session.

    return render_template(
        "dashboard.html",
        stats=stats_html,
        heatmap_filename=heatmap_file,
        importance_filename=importance_filename,
        chart_data=None,
        chart_labels=None,
    )

if __name__ == "__main__":
    app.run(debug=True)
