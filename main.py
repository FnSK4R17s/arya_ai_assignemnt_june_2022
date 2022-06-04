import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import sweetviz as sv
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

VIS = "visualizations"


def visualize(
    train_data: pd.DataFrame, test_data: pd.DataFrame, train_features: pd.DataFrame,
):
    print(train_data.head())
    print(train_data.describe())

    report: sv.DataframeReport = sv.analyze(train_data.reset_index(drop=True))
    report.show_html(VIS + "/train_set_analysis.html", open_browser=False)

    comparitive_analysis: sv.DataframeReport = sv.compare(
        train_features.reset_index(drop=True), test_data.reset_index(drop=True)
    )
    comparitive_analysis.show_html(
        VIS + "/comparitive_analysis.html", open_browser=False
    )


def main():

    train_data = pd.read_csv("datasets/training_set.csv", index_col=0)
    test_data = pd.read_csv("datasets/test_set.csv", index_col=0)

    train_targets = train_data["Y"]
    train_features = train_data.drop(["Y"], axis=1)

    # visualize(train_data, test_data, train_features)
    train_model(train_targets, train_features, 800)

    filtered_train_data = filter_data(train_data)

    train_targets = filtered_train_data["Y"]
    train_features = filtered_train_data.drop(["Y"], axis=1)

    train_model(train_targets, train_features, 100)

    ANOVA = f_classif(train_data.drop(["Y"], axis=1), train_data["Y"])

    print(ANOVA)

    X_new = SelectKBest(f_classif, k=10).fit_transform(
        train_data.drop(["Y"], axis=1), train_data["Y"]
    )

    print(X_new.shape)

    train_model(train_data["Y"], X_new, 1000)

    X_new = SelectKBest(kendalls_all, k=10).fit_transform(
        train_data.drop(["Y"], axis=1), train_data["Y"]
    )

    print(X_new.shape)

    train_model(train_data["Y"], X_new, 1000)


def filter_data(train_data):
    kds = {}
    for col in train_data.columns:
        if "col" != "Y":
            corr = kendalls(train_data[col], train_data["Y"])
            kds.update({col: corr})

    print(kds)

    fig_dims = (12, 9)
    fig, ax = plt.subplots(figsize=fig_dims)
    corr = train_data.corr()
    hm = sns.heatmap(corr, ax=ax)
    plt.savefig(
        VIS + "/heatmap.png", bbox_inches="tight", dpi=300,
    )
    correlations = abs(corr["Y"])[abs(corr["Y"]) > 0.1].sort_values(ascending=False)
    print(correlations)

    selected = train_data[correlations.index]
    hm = sns.heatmap(selected.corr(), ax=ax)
    plt.savefig(
        VIS + "/heatmap_selected.png", bbox_inches="tight", dpi=300,
    )

    # dropping the X26
    # selected = selected.drop(["X26"], axis=1)
    # hm = sns.heatmap(selected.corr(), ax=ax)
    # plt.savefig(
    #     VIS + "/heatmap_selected_final.png", bbox_inches="tight", dpi=300,
    # )
    return selected


def kendalls(x, y) -> float:
    """
    Kendall’s rank coefficient is a measure of the similarity between two rankings.
    """
    corr, _ = kendalltau(x, y)
    # print("Kendall Rank correlation: %.5f" % corr)
    return corr


def kendalls_all(X, y) -> float:
    """
    Kendall’s rank coefficient is a measure of the similarity between two rankings.
    """
    corrs = []
    ps = []
    for col in X.T:
        corr, p = kendalltau(col, y)
        corrs.append(corr)
        ps.append(p)

    return [corrs, ps]


def train_model(train_targets, train_features, iter):
    X_train, X_test, y_train, y_test = train_test_split(
        train_features, train_targets, test_size=0.2, random_state=42
    )

    # pipe = make_pipeline(StandardScaler(), LogisticRegression())
    # pipe.fit(X_train, y_train)
    # score = pipe.score(X_test, y_test)

    # print(score)
    scaler = StandardScaler().fit(X_train)
    X_scaled = scaler.transform(X_train)

    model = LogisticRegression(max_iter=iter)
    model.fit(X_scaled, y_train)

    scaler = StandardScaler().fit(X_test)
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
