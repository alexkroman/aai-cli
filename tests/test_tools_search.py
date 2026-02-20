"""Tests for tools_search.py — dataset search and API spec tools."""

from unittest.mock import MagicMock, patch

from aai_cli.tools_search import get_dataset_info, search_assemblyai_api, search_audio_datasets

# ---------------------------------------------------------------------------
# search_audio_datasets
# ---------------------------------------------------------------------------


@patch("huggingface_hub.HfApi")
def test_search_audio_datasets_found(mock_hf_cls):
    mock_api = MagicMock()
    mock_ds = MagicMock()
    mock_ds.id = "test/dataset"
    mock_ds.downloads = 1000
    mock_ds.description = "A test dataset"
    mock_ds.pretty_name = "Test Dataset"
    mock_ds.languages = ["en"]
    mock_ds.card_data = None
    mock_api.list_datasets.return_value = [mock_ds]
    mock_hf_cls.return_value = mock_api

    result = search_audio_datasets("test")
    assert "test/dataset" in result
    assert "1,000 downloads" in result


@patch("huggingface_hub.HfApi")
def test_search_audio_datasets_empty(mock_hf_cls):
    mock_api = MagicMock()
    mock_api.list_datasets.return_value = []
    mock_hf_cls.return_value = mock_api

    result = search_audio_datasets("nonexistent_query_xyz")
    assert "No datasets found" in result


@patch("huggingface_hub.HfApi")
def test_search_audio_datasets_with_language(mock_hf_cls):
    mock_api = MagicMock()
    mock_api.list_datasets.return_value = []
    mock_hf_cls.return_value = mock_api

    search_audio_datasets("test", language="fr")
    call_kwargs = mock_api.list_datasets.call_args[1]
    assert call_kwargs["language"] == "fr"


# ---------------------------------------------------------------------------
# get_dataset_info
# ---------------------------------------------------------------------------


@patch("datasets.get_dataset_infos")
@patch("datasets.get_dataset_config_names")
def test_get_dataset_info(mock_configs, mock_infos):
    mock_configs.return_value = ["default", "en"]
    mock_info = MagicMock()
    mock_info.features = {"audio": MagicMock(), "text": MagicMock()}
    mock_split = MagicMock()
    mock_split.num_examples = 100
    mock_info.splits = {"test": mock_split}
    mock_infos.return_value = {"default": mock_info}

    result = get_dataset_info("test/dataset")
    assert "test/dataset" in result
    assert "default" in result
    assert "100" in result


# ---------------------------------------------------------------------------
# search_assemblyai_api
# ---------------------------------------------------------------------------


@patch("aai_cli.tools_search._fetch_openapi_spec")
def test_search_assemblyai_api_found(mock_fetch):
    mock_fetch.return_value = (
        "paths:\n  /v2/transcript:\n    post:\n      summary: Create transcript\n"
    )
    result = search_assemblyai_api("transcript")
    assert "transcript" in result.lower()
    assert "match" in result.lower()


@patch("aai_cli.tools_search._fetch_openapi_spec")
def test_search_assemblyai_api_no_match(mock_fetch):
    mock_fetch.return_value = "openapi: 3.0.0\ninfo:\n  title: AssemblyAI\n"
    result = search_assemblyai_api("xyznonexistent")
    assert "No matches" in result


@patch("aai_cli.tools_search._fetch_openapi_spec")
def test_search_assemblyai_api_empty_query(mock_fetch):
    mock_fetch.return_value = "some spec"
    result = search_assemblyai_api("")
    assert "No query" in result


@patch("aai_cli.tools_search._fetch_openapi_spec")
def test_search_assemblyai_api_error(mock_fetch):
    mock_fetch.side_effect = RuntimeError("network error")
    result = search_assemblyai_api("test")
    assert "Error" in result
