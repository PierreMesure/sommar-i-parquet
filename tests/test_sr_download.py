from src.sr.download import EXCLUDED_IMAGE_URLS, download_episode_portraits


def test_download_episode_portraits_replaces_excluded_image(monkeypatch) -> None:
    excluded_url = next(iter(EXCLUDED_IMAGE_URLS))
    episodes = [
        {"id": 2815552, "imageurltemplate": excluded_url},
        {"id": 1, "imageurltemplate": "https://example.test/portrait.jpg"},
    ]

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"item": {"imageSrc": "https://example.test/niklas.jpg"}}

    class Client:
        def __init__(self, **kwargs):
            self.calls = []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, url, params):
            self.calls.append((url, params))
            return Response()

    monkeypatch.setattr("src.sr.download.httpx.Client", Client)

    result = download_episode_portraits(episodes)

    assert result[0]["imageurltemplate"] == "https://example.test/niklas.jpg"
    assert result[0]["imageurl"] == "https://example.test/niklas.jpg"
    assert result[1]["imageurltemplate"] == "https://example.test/portrait.jpg"
