from recommonmark.transform import AutoStructify


def setup(app):
    app.add_transform(AutoStructify)
