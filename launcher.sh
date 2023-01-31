export FLASK_APP="src/wine_predictor_api:create_app"
export FLASK_DEBUG=true
export API_CONFIG="config.json"
PORT=5000

python -m flask run -h 0.0.0.0 -p $PORT
