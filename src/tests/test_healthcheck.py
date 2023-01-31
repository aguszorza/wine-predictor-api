def test_ping(app_client):
    # given
    endpoint = "/ping"

    # when
    response = app_client.get(endpoint)

    # then
    assert response.status_code == 200
    assert "pong" in response.text
