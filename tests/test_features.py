import pytest
from shapely.geometry import LineString, Point, Polygon

from toposlicr.config import (
    LakesConfig,
    PeaksConfig,
    RiversConfig,
    SymbologyConfig,
)
from toposlicr.features.overpass import parse_overpass_json
from toposlicr.features.schema import FeatureCollection, FeatureType, GeoFeature
from toposlicr.features.select import (
    auto_choices,
    read_features_csv,
    resolve_features,
    write_features_csv,
)

# --- Overpass JSON parsing -------------------------------------------------


def _way_ring():
    # A small closed square around lat 36.5, lon -118.3 (first == last).
    return [
        {"lat": 36.50, "lon": -118.30},
        {"lat": 36.50, "lon": -118.29},
        {"lat": 36.51, "lon": -118.29},
        {"lat": 36.51, "lon": -118.30},
        {"lat": 36.50, "lon": -118.30},
    ]


def test_parse_peak_node():
    payload = {"elements": [{
        "type": "node", "id": 1, "lat": 36.578, "lon": -118.292,
        "tags": {"natural": "peak", "name": "Mount Whitney", "ele": "4421"},
    }]}
    coll = parse_overpass_json(payload)
    assert len(coll) == 1
    f = coll.features[0]
    assert f.feature_type is FeatureType.PEAK
    assert f.name == "Mount Whitney"
    assert f.elevation == pytest.approx(4421.0)
    assert f.importance == pytest.approx(4421.0)
    assert isinstance(f.geometry, Point)
    assert f.osm_id == "node/1"


def test_parse_place_node():
    payload = {"elements": [{
        "type": "node", "id": 2, "lat": 36.6, "lon": -118.06,
        "tags": {"place": "village", "name": "Lone Pine"},
    }]}
    coll = parse_overpass_json(payload)
    assert len(coll) == 1
    f = coll.features[0]
    assert f.feature_type is FeatureType.PLACE
    assert f.name == "Lone Pine"


def test_parse_lake_way():
    payload = {"elements": [{
        "type": "way", "id": 3, "geometry": _way_ring(),
        "tags": {"natural": "water", "name": "Iceberg Lake"},
    }]}
    coll = parse_overpass_json(payload)
    assert len(coll) == 1
    f = coll.features[0]
    assert f.feature_type is FeatureType.LAKE
    assert f.name == "Iceberg Lake"
    assert isinstance(f.geometry, Polygon)
    # importance is area in km², positive and small for this ~1 km box.
    assert f.importance is not None and f.importance > 0
    assert f.importance < 5


def test_parse_river_way_importance_five():
    payload = {"elements": [{
        "type": "way", "id": 4,
        "geometry": [{"lat": 36.5, "lon": -118.3}, {"lat": 36.51, "lon": -118.29}],
        "tags": {"waterway": "river", "name": "Big River"},
    }]}
    f = parse_overpass_json(payload).features[0]
    assert f.feature_type is FeatureType.RIVER
    assert isinstance(f.geometry, LineString)
    assert f.importance == 5


def test_parse_stream_way_importance_two():
    payload = {"elements": [{
        "type": "way", "id": 5,
        "geometry": [{"lat": 36.5, "lon": -118.3}, {"lat": 36.51, "lon": -118.29}],
        "tags": {"waterway": "stream"},
    }]}
    f = parse_overpass_json(payload).features[0]
    assert f.feature_type is FeatureType.RIVER
    assert f.importance == 2


def test_parse_mixed_payload_counts():
    payload = {"elements": [
        {"type": "node", "id": 1, "lat": 36.578, "lon": -118.292,
         "tags": {"natural": "peak", "name": "Peak", "ele": "4000"}},
        {"type": "way", "id": 3, "geometry": _way_ring(),
         "tags": {"natural": "water", "name": "Lake"}},
        {"type": "way", "id": 4,
         "geometry": [{"lat": 36.5, "lon": -118.3}, {"lat": 36.51, "lon": -118.29}],
         "tags": {"waterway": "stream"}},
    ]}
    coll = parse_overpass_json(payload)
    assert len(coll.of_type(FeatureType.PEAK)) == 1
    assert len(coll.of_type(FeatureType.LAKE)) == 1
    assert len(coll.of_type(FeatureType.RIVER)) == 1


# --- Feature selection & the features.csv loop -----------------------------


def _peak(name, ele, osm_id):
    return GeoFeature(FeatureType.PEAK, Point(-118.3, 36.5), name=name,
                      elevation=ele, importance=ele, osm_id=osm_id)


def _lake(area_km2, osm_id, name="Lake"):
    return GeoFeature(FeatureType.LAKE, Polygon(_lake_ring()), name=name,
                      importance=area_km2, osm_id=osm_id)


def _lake_ring():
    return [(-118.30, 36.50), (-118.29, 36.50), (-118.29, 36.51),
            (-118.30, 36.51), (-118.30, 36.50)]


def _river(order, osm_id, name="River"):
    return GeoFeature(FeatureType.RIVER,
                      LineString([(-118.30, 36.50), (-118.29, 36.51)]),
                      name=name, importance=order, osm_id=osm_id)


def _by_id(choices):
    return {c.feature.osm_id: c for c in choices}


def test_auto_choices_applies_river_stream_order():
    coll = FeatureCollection(features=[
        _river(5, "way/1", "Main River"),
        _river(2, "way/2", "Small Stream"),
    ])
    sym = SymbologyConfig(rivers=RiversConfig(min_stream_order=3))
    by_id = _by_id(auto_choices(coll, sym))
    assert by_id["way/1"].include is True
    assert by_id["way/2"].include is False


def test_auto_choices_applies_lake_area_threshold():
    coll = FeatureCollection(features=[
        _lake(0.10, "way/1", "Big Lake"),
        _lake(0.01, "way/2", "Puddle"),
    ])
    sym = SymbologyConfig(lakes=LakesConfig(min_area_km2=0.05))
    by_id = _by_id(auto_choices(coll, sym))
    assert by_id["way/1"].include is True
    assert by_id["way/2"].include is False


def test_auto_choices_named_peak_included_unnamed_excluded():
    coll = FeatureCollection(features=[
        _peak("Mount Whitney", 4421, "node/1"),
        GeoFeature(FeatureType.PEAK, Point(-118.3, 36.5), name=None,
                   elevation=4000, importance=4000, osm_id="node/2"),
    ])
    sym = SymbologyConfig()
    by_id = _by_id(auto_choices(coll, sym))
    assert by_id["node/1"].include is True
    assert by_id["node/2"].include is False


def test_peaks_max_count_keeps_highest():
    coll = FeatureCollection(features=[
        _peak("A", 4400, "node/1"),
        _peak("B", 4200, "node/2"),
        _peak("C", 4300, "node/3"),
    ])
    sym = SymbologyConfig(peaks=PeaksConfig(max_count=2))
    by_id = _by_id(auto_choices(coll, sym))
    # Two highest (4400, 4300) kept; 4200 dropped.
    assert by_id["node/1"].include is True
    assert by_id["node/3"].include is True
    assert by_id["node/2"].include is False


def test_resolve_features_writes_csv_first_run(tmp_path):
    coll = FeatureCollection(features=[
        _peak("Mount Whitney", 4421, "node/1"),
        _river(5, "way/1", "Main River"),
    ])
    csv_path = tmp_path / "features.csv"
    kept, existed = resolve_features(coll, SymbologyConfig(), csv_path)
    assert existed is False
    assert csv_path.is_file()
    # Both features pass default thresholds.
    ids = {f.osm_id for f in kept.features}
    assert ids == {"node/1", "way/1"}


def test_resolve_features_second_run_honors_edits(tmp_path):
    coll = FeatureCollection(features=[
        _peak("Mount Whitney", 4421, "node/1"),
        _peak("Lone Peak", 4000, "node/2"),
    ])
    csv_path = tmp_path / "features.csv"
    # First run writes the CSV.
    resolve_features(coll, SymbologyConfig(), csv_path)

    # Hand-edit: exclude node/2, override node/1's label.
    choices = auto_choices(coll, SymbologyConfig())
    by_id = _by_id(choices)
    by_id["node/1"].label = "Whitney"
    by_id["node/2"].include = False
    write_features_csv(choices, csv_path)

    # Fresh features (new objects) so overrides are applied by osm_id, not identity.
    coll2 = FeatureCollection(features=[
        _peak("Mount Whitney", 4421, "node/1"),
        _peak("Lone Peak", 4000, "node/2"),
    ])
    kept, existed = resolve_features(coll2, SymbologyConfig(), csv_path)
    assert existed is True
    ids = {f.osm_id for f in kept.features}
    assert ids == {"node/1"}
    # Label override replaced the feature name in the kept collection.
    kept_feature = kept.features[0]
    assert kept_feature.name == "Whitney"


def test_gui_overrides_win_over_existing_csv(tmp_path):
    """A fresh GUI selection must beat a stale features.csv (precedence)."""
    coll = FeatureCollection(features=[
        _peak("Mount Whitney", 4421, "node/1"),
        _peak("Lone Peak", 4000, "node/2"),
    ])
    csv_path = tmp_path / "features.csv"
    # An existing CSV excludes node/1.
    choices = auto_choices(coll, SymbologyConfig())
    _by_id(choices)["node/1"].include = False
    write_features_csv(choices, csv_path)

    coll2 = FeatureCollection(features=[
        _peak("Mount Whitney", 4421, "node/1"),
        _peak("Lone Peak", 4000, "node/2"),
    ])
    # GUI now re-includes node/1 and excludes node/2 — the GUI wins over the CSV.
    kept, _ = resolve_features(coll2, SymbologyConfig(), csv_path,
                               overrides={"node/1": True, "node/2": False})
    assert {f.osm_id for f in kept.features} == {"node/1"}


def test_read_features_csv_roundtrip(tmp_path):
    coll = FeatureCollection(features=[_peak("Mount Whitney", 4421, "node/1")])
    choices = auto_choices(coll, SymbologyConfig())
    csv_path = tmp_path / "features.csv"
    write_features_csv(choices, csv_path)
    overrides = read_features_csv(csv_path)
    assert "node/1" in overrides
    include, label = overrides["node/1"]
    assert include is True
    assert label == "Mount Whitney"
