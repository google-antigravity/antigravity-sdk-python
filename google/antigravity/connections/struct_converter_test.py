# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests verifying bidirectional conversion between Python dicts and genai Content Struct protos."""

import dataclasses
import enum
from typing import Any

from absl.testing import absltest
import pydantic

from google.antigravity.proto import content_pb2
from google.antigravity import types
from google.antigravity.connections import struct_converter


class SamplePydanticModel(pydantic.BaseModel):
  model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
  output: str
  video: Any


@dataclasses.dataclass
class SampleDataclass:
  output: str
  video: Any


class StructConverterTest(absltest.TestCase):

  def test_is_structured(self):
    self.assertTrue(struct_converter.is_structured({"a": 1}))
    self.assertTrue(
        struct_converter.is_structured(
            SamplePydanticModel(output="test", video=None)
        )
    )
    self.assertTrue(
        struct_converter.is_structured(
            SampleDataclass(output="test", video=None)
        )
    )
    self.assertFalse(struct_converter.is_structured("string"))
    self.assertFalse(struct_converter.is_structured(123))
    self.assertFalse(struct_converter.is_structured(True))
    self.assertFalse(struct_converter.is_structured(None))
    self.assertFalse(struct_converter.is_structured([1, 2, 3]))
    self.assertFalse(struct_converter.is_structured(SampleDataclass))
    self.assertFalse(struct_converter.is_structured(SamplePydanticModel))
    self.assertFalse(
        struct_converter.is_structured(
            content_pb2.VideoContent(uri="https://vid")
        )
    )
    self.assertFalse(struct_converter.is_structured(content_pb2.Struct()))

  def test_to_struct_passthrough_struct(self):
    struct_in = content_pb2.Struct(
        fields=[
            content_pb2.Field(
                name="x", value=content_pb2.Value(number_value=1.0)
            )
        ]
    )
    res = struct_converter.to_struct(struct_in)
    self.assertIs(res, struct_in)

  def test_value_to_proto_primitives_and_protos(self):
    # Test bool before int/float
    val_true = struct_converter.value_to_proto(True)
    self.assertTrue(val_true.HasField("bool_value"))
    self.assertFalse(val_true.HasField("number_value"))
    self.assertTrue(val_true.bool_value)

    val_false = struct_converter.value_to_proto(False)
    self.assertTrue(val_false.HasField("bool_value"))
    self.assertFalse(val_false.HasField("number_value"))
    self.assertFalse(val_false.bool_value)

    # Test None
    val_none = struct_converter.value_to_proto(None)
    self.assertTrue(val_none.HasField("null_value"))
    self.assertEqual(val_none.null_value, 0)

    # Test float and int
    val_float = struct_converter.value_to_proto(3.14)
    self.assertTrue(val_float.HasField("number_value"))
    self.assertAlmostEqual(val_float.number_value, 3.14)

    val_int = struct_converter.value_to_proto(42)
    self.assertTrue(val_int.HasField("number_value"))
    self.assertEqual(val_int.number_value, 42.0)

    # Test str
    val_str = struct_converter.value_to_proto("hello")
    self.assertEqual(val_str.string_value, "hello")

    # Test TextContent
    txt = content_pb2.TextContent(text="sample text")
    val_txt = struct_converter.value_to_proto(txt)
    self.assertEqual(val_txt.content_value.text.text, "sample text")

    # Test Content
    cnt = content_pb2.Content(text=content_pb2.TextContent(text="hi"))
    val_cnt = struct_converter.value_to_proto(cnt)
    self.assertEqual(val_cnt.content_value.text.text, "hi")

    # Test Struct
    st = content_pb2.Struct(
        fields=[
            content_pb2.Field(
                name="k", value=content_pb2.Value(string_value="v")
            )
        ]
    )
    val_st = struct_converter.value_to_proto(st)
    self.assertEqual(val_st.struct_value.fields[0].value.string_value, "v")

    # Test ListValue
    lv = content_pb2.ListValue(values=[content_pb2.Value(string_value="item")])
    val_lv = struct_converter.value_to_proto(lv)
    self.assertEqual(val_lv.list_value.values[0].string_value, "item")

    # Test Value direct passthrough
    v = content_pb2.Value(string_value="direct")
    val_v = struct_converter.value_to_proto(v)
    self.assertIs(val_v, v)

  def test_dict_to_struct_video_content(self):
    video = content_pb2.VideoContent(
        mime_type=content_pb2.VideoContent.TYPE_MP4,
        uri="https://example.com/video/123",
    )
    data = {"output": "done", "video": video}
    struct_pb = struct_converter.dict_to_struct(data)
    self.assertLen(struct_pb.fields, 2)
    fields_map = {f.name: f.value for f in struct_pb.fields}
    self.assertEqual(fields_map["output"].string_value, "done")
    self.assertTrue(fields_map["video"].HasField("content_value"))
    self.assertEqual(
        fields_map["video"].content_value.video.uri,
        "https://example.com/video/123",
    )

  def test_dict_to_struct_preserves_order(self):
    data = {"z": 1, "a": 2, "m": 3}
    struct_pb = struct_converter.dict_to_struct(data)
    field_names = [f.name for f in struct_pb.fields]
    self.assertEqual(field_names, ["z", "a", "m"])

  def test_to_struct_pydantic_model(self):
    video = content_pb2.VideoContent(
        mime_type=content_pb2.VideoContent.TYPE_MP4,
        uri="https://example.com/video/123",
    )
    model = SamplePydanticModel(output="done", video=video)
    struct_pb = struct_converter.to_struct(model)
    fields_map = {f.name: f.value for f in struct_pb.fields}
    self.assertEqual(fields_map["output"].string_value, "done")
    self.assertTrue(fields_map["video"].HasField("content_value"))
    self.assertEqual(
        fields_map["video"].content_value.video.uri,
        "https://example.com/video/123",
    )

  def test_to_struct_dataclass(self):
    video = content_pb2.VideoContent(
        mime_type=content_pb2.VideoContent.TYPE_MP4,
        uri="https://example.com/video/123",
    )
    dc = SampleDataclass(output="done", video=video)
    struct_pb = struct_converter.to_struct(dc)
    fields_map = {f.name: f.value for f in struct_pb.fields}
    self.assertEqual(fields_map["output"].string_value, "done")
    self.assertTrue(fields_map["video"].HasField("content_value"))
    self.assertEqual(
        fields_map["video"].content_value.video.uri,
        "https://example.com/video/123",
    )

  def test_value_to_proto_nested_pydantic_model(self):
    video = content_pb2.VideoContent(
        mime_type=content_pb2.VideoContent.TYPE_MP4,
        uri="https://example.com/video/nested",
    )
    model = SamplePydanticModel(output="nested_done", video=video)
    data = {"wrapper": model}
    struct_pb = struct_converter.dict_to_struct(data)
    fields_map = {f.name: f.value for f in struct_pb.fields}
    wrapper_val = fields_map["wrapper"]
    self.assertTrue(wrapper_val.HasField("struct_value"))
    wrapper_fields = {f.name: f.value for f in wrapper_val.struct_value.fields}
    self.assertEqual(wrapper_fields["output"].string_value, "nested_done")
    self.assertEqual(
        wrapper_fields["video"].content_value.video.uri,
        "https://example.com/video/nested",
    )

  def test_to_json_fallback_with_display_name(self):
    class DummyMedia:
      DESCRIPTOR = type("Desc", (), {"name": "ImageContent"})()
      display_name = "img_123"

    image = DummyMedia()
    data = {"output": "done", "image": image}
    fallback = struct_converter.to_json_fallback(data)
    self.assertEqual(fallback, {"output": "done", "image": {"$ref": "img_123"}})

  def test_to_json_fallback_with_uri(self):
    video = content_pb2.VideoContent(
        mime_type=content_pb2.VideoContent.TYPE_MP4,
        uri="https://example.com/video/123",
    )
    data = {"output": "done", "video": video}
    fallback = struct_converter.to_json_fallback(data)
    self.assertEqual(
        fallback,
        {"output": "done", "video": {"$ref": "https://example.com/video/123"}},
    )

  def test_to_json_fallback_pydantic_model(self):
    video = content_pb2.VideoContent(
        mime_type=content_pb2.VideoContent.TYPE_MP4,
        uri="https://example.com/video/123",
    )
    model = SamplePydanticModel(output="done", video=video)
    fallback = struct_converter.to_json_fallback(model)
    self.assertEqual(
        fallback,
        {"output": "done", "video": {"$ref": "https://example.com/video/123"}},
    )

  def test_dict_to_struct_nested_structs_and_lists(self):
    video = content_pb2.VideoContent(
        mime_type=content_pb2.VideoContent.TYPE_MP4,
        uri="https://example.com/video/nested_123",
    )
    image = content_pb2.ImageContent(
        mime_type=content_pb2.ImageContent.TYPE_JPEG,
        uri="http://example.com/img.jpg",
    )

    data = {
        "status": "ok",
        "conversation": {
            "media_list": [video, image],
            "metadata": {"step": 1},
        },
    }

    struct_pb = struct_converter.dict_to_struct(data)
    fields_map = {f.name: f.value for f in struct_pb.fields}

    conv_struct = fields_map["conversation"].struct_value
    conv_fields = {f.name: f.value for f in conv_struct.fields}
    media_list = conv_fields["media_list"].list_value.values

    self.assertLen(media_list, 2)
    vid_val = media_list[0].content_value.video
    self.assertEqual(vid_val.uri, "https://example.com/video/nested_123")

    img_val = media_list[1].content_value.image
    self.assertEqual(img_val.uri, "http://example.com/img.jpg")

    fallback = struct_converter.to_json_fallback(data)
    self.assertEqual(
        fallback,
        {
            "status": "ok",
            "conversation": {
                "media_list": [
                    {"$ref": "https://example.com/video/nested_123"},
                    {"$ref": "http://example.com/img.jpg"},
                ],
                "metadata": {"step": 1},
            },
        },
    )

  def test_dict_to_struct_all_media_content_types(self):
    video = content_pb2.VideoContent(uri="vid_uri")
    audio = content_pb2.AudioContent(uri="aud_uri")
    image = content_pb2.ImageContent(uri="img_uri")
    document = content_pb2.DocumentContent(uri="doc_uri")

    data = {"v": video, "a": audio, "i": image, "d": document}
    struct_pb = struct_converter.dict_to_struct(data)
    fields_map = {f.name: f.value for f in struct_pb.fields}

    self.assertEqual(fields_map["v"].content_value.video.uri, "vid_uri")
    self.assertEqual(fields_map["a"].content_value.audio.uri, "aud_uri")
    self.assertEqual(fields_map["i"].content_value.image.uri, "img_uri")
    self.assertEqual(fields_map["d"].content_value.document.uri, "doc_uri")

  def test_dynamic_content_proto_discovery(self):
    """Verifies that Content proto types and descriptor names are dynamically discovered."""
    self.assertIn(content_pb2.Content, struct_converter._CONTENT_PROTO_TYPES)
    self.assertIn(
        content_pb2.VideoContent, struct_converter._CONTENT_PROTO_TYPES
    )
    self.assertIn(
        content_pb2.AudioContent, struct_converter._CONTENT_PROTO_TYPES
    )
    self.assertIn(
        content_pb2.ImageContent, struct_converter._CONTENT_PROTO_TYPES
    )
    self.assertIn(
        content_pb2.DocumentContent, struct_converter._CONTENT_PROTO_TYPES
    )
    self.assertIn(
        content_pb2.TextContent, struct_converter._CONTENT_PROTO_TYPES
    )
    self.assertIn("Content", struct_converter._CONTENT_DESCRIPTOR_NAMES)
    self.assertIn("VideoContent", struct_converter._CONTENT_DESCRIPTOR_NAMES)
    self.assertNotIn("Struct", struct_converter._CONTENT_DESCRIPTOR_NAMES)
    self.assertNotIn("Value", struct_converter._CONTENT_DESCRIPTOR_NAMES)

  def test_to_json_fallback_struct_listvalue_value_and_unrecognized(self):
    struct_pb = content_pb2.Struct(
        fields=[
            content_pb2.Field(
                name="msg", value=content_pb2.Value(string_value="hello")
            ),
            content_pb2.Field(
                name="video",
                value=content_pb2.Value(
                    content_value=content_pb2.Content(
                        video=content_pb2.VideoContent(
                            uri="https://example.com/vid"
                        )
                    )
                ),
            ),
        ]
    )
    self.assertEqual(
        struct_converter.to_json_fallback(struct_pb),
        {"msg": "hello", "video": {"$ref": "https://example.com/vid"}},
    )

    list_pb = content_pb2.ListValue(
        values=[
            content_pb2.Value(number_value=1.5),
            content_pb2.Value(bool_value=True),
        ]
    )
    self.assertEqual(struct_converter.to_json_fallback(list_pb), [1.5, True])

    val_null = content_pb2.Value(null_value="NULL_VALUE")
    self.assertIsNone(struct_converter.to_json_fallback(val_null))

    val_bool = content_pb2.Value(bool_value=False)
    self.assertFalse(struct_converter.to_json_fallback(val_bool))

    txt = content_pb2.TextContent(text="inline text")
    self.assertEqual(struct_converter.to_json_fallback(txt), "inline text")

    cnt_txt = content_pb2.Content(text=content_pb2.TextContent(text="wrapped"))
    self.assertEqual(struct_converter.to_json_fallback(cnt_txt), "wrapped")

    prim_struct = struct_converter.to_struct("raw")
    self.assertEqual(prim_struct.fields[0].name, "result")
    self.assertEqual(prim_struct.fields[0].value.string_value, "raw")

    wrapped_uri = content_pb2.Content(
        image=content_pb2.ImageContent(uri="https://example.com/wrapped.png")
    )
    self.assertEqual(
        struct_converter.to_json_fallback(wrapped_uri),
        {"$ref": "https://example.com/wrapped.png"},
    )
    self.assertEqual(
        struct_converter.to_json_fallback(
            content_pb2.Value(struct_value=prim_struct)
        ),
        {"result": "raw"},
    )
    self.assertEqual(
        struct_converter.to_json_fallback(
            content_pb2.Value(
                list_value=content_pb2.ListValue(
                    values=[content_pb2.Value(string_value="x")]
                )
            )
        ),
        ["x"],
    )
    self.assertEqual(struct_converter.to_json_fallback(content_pb2.Value()), "")

    class UnserializableCustomObj:

      def __str__(self):
        return "custom_str_repr"

    custom_obj = UnserializableCustomObj()
    fallback = struct_converter.to_json_fallback(custom_obj)
    self.assertEqual(fallback, "custom_str_repr")

  def test_unwrap_wire_struct_and_wire_value(self):
    # Test flat wire struct with various value types
    wire_dict = {
        "fields": [
            {"name": "str_key", "value": {"string_value": "hello"}},
            {"name": "num_key", "value": {"number_value": 42.0}},
            {"name": "bool_key", "value": {"bool_value": True}},
            {"name": "null_key", "value": {"null_value": "NULL_VALUE"}},
            {
                "name": "nested_struct",
                "value": {
                    "struct_value": {
                        "fields": [
                            {
                                "name": "inner_k",
                                "value": {"stringValue": "nested"},
                            },
                            {
                                "name": "inner_num",
                                "value": {"numberValue": 10},
                            },
                            {
                                "name": "inner_bool",
                                "value": {"boolValue": False},
                            },
                            {
                                "name": "inner_null",
                                "value": {"nullValue": 0},
                            },
                        ]
                    }
                },
            },
            {
                "name": "list_key",
                "value": {
                    "list_value": {
                        "values": [
                            {"string_value": "item1"},
                            {"number_value": 2.0},
                        ]
                    }
                },
            },
            {
                "name": "content_key",
                "value": {"content_value": {"text": {"text": "txt"}}},
            },
        ]
    }
    unpacked = struct_converter.unwrap_wire_struct(wire_dict)
    self.assertEqual(
        unpacked,
        {
            "str_key": "hello",
            "num_key": 42.0,
            "bool_key": True,
            "null_key": None,
            "nested_struct": {
                "inner_k": "nested",
                "inner_num": 10,
                "inner_bool": False,
                "inner_null": None,
            },
            "list_key": ["item1", 2.0],
            "content_key": {"text": {"text": "txt"}},
        },
    )

    # Test passthrough for already plain dict
    plain = {"a": 1, "b": "c"}
    self.assertEqual(struct_converter.unwrap_wire_struct(plain), plain)

    # Test non-dict inputs
    self.assertEqual(struct_converter.unwrap_wire_struct(None), {})
    self.assertEqual(struct_converter.unwrap_wire_struct("string"), {})
    self.assertEqual(struct_converter.unwrap_wire_struct([1, 2]), {})

    # Test unwrap_wire_value non-dict passthrough
    self.assertEqual(
        struct_converter.unwrap_wire_value("raw_string"), "raw_string"
    )
    self.assertEqual(struct_converter.unwrap_wire_value(123), 123)

  def test_sdk_media_types_conversion(self):
    cases = [
        (types.Image(data=b"\x89PNG", mime_type="image/png"), "image"),
        (types.Audio(data=b"RIFF", mime_type="audio/wav"), "audio"),
        (types.Video(data=b"ftyp", mime_type="video/mp4"), "video"),
        (types.Document(data=b"%PDF", mime_type="application/pdf"), "document"),
    ]
    for media_obj, field_name in cases:
      with self.subTest(field_name):
        self.assertFalse(struct_converter.is_structured(media_obj))
        val_pb = struct_converter.value_to_proto(media_obj)
        self.assertEqual(val_pb.content_value.WhichOneof("type"), field_name)
        inner = getattr(val_pb.content_value, field_name)
        self.assertEqual(inner.data, media_obj.data)
        self.assertEqual(inner.mime_type_string, media_obj.mime_type)

    top_struct = struct_converter.to_struct(cases[0][0])
    self.assertEqual(top_struct.fields[0].name, "result")
    self.assertEqual(
        top_struct.fields[0].value.content_value.image.data, b"\x89PNG"
    )

    nested = SamplePydanticModel(output="ok", video=cases[2][0])
    nested_fields = {
        f.name: f.value for f in struct_converter.to_struct(nested).fields
    }
    self.assertEqual(nested_fields["output"].string_value, "ok")
    self.assertEqual(nested_fields["video"].content_value.video.data, b"ftyp")
    self.assertEqual(
        nested_fields["video"].content_value.video.mime_type_string, "video/mp4"
    )

  def test_pydantic_base_model_preserves_exclude_computed_extra_and_enums(self):
    class ReportStatus(enum.Enum):
      OK = "ok"

    class PlainRedactedReport(pydantic.BaseModel):
      model_config = pydantic.ConfigDict(extra="allow")
      summary: str
      status: ReportStatus = ReportStatus.OK
      api_token: str = pydantic.Field(default="secret-token", exclude=True)

      @pydantic.computed_field
      @property
      def label(self) -> str:
        return f"{self.summary}:{self.status.value}"

    plain_report = PlainRedactedReport(
        summary="done", extra_note="kept"  # type: ignore[call-arg]
    )
    plain_fields = {
        f.name: f.value for f in struct_converter.to_struct(plain_report).fields
    }
    self.assertNotIn("api_token", plain_fields)
    self.assertEqual(plain_fields["summary"].string_value, "done")
    self.assertEqual(plain_fields["status"].string_value, "ok")
    self.assertEqual(plain_fields["label"].string_value, "done:ok")
    self.assertEqual(plain_fields["extra_note"].string_value, "kept")
    self.assertEqual(
        struct_converter.to_json_fallback(plain_report),
        {
            "summary": "done",
            "status": "ok",
            "label": "done:ok",
            "extra_note": "kept",
        },
    )

    class MediaRedactedReport(pydantic.BaseModel):
      model_config = pydantic.ConfigDict(
          arbitrary_types_allowed=True, extra="allow"
      )
      summary: str
      status: ReportStatus = ReportStatus.OK
      image: types.Image
      api_token: str = pydantic.Field(default="secret-token", exclude=True)

      @pydantic.computed_field
      @property
      def label(self) -> str:
        return f"{self.summary}:{self.status.value}"

    media_report = MediaRedactedReport(
        summary="done",
        image=types.Image(data=b"\x89PNG", mime_type="image/png"),
        extra_note="kept",  # type: ignore[call-arg]
    )
    media_fields = {
        f.name: f.value for f in struct_converter.to_struct(media_report).fields
    }
    self.assertNotIn("api_token", media_fields)
    self.assertEqual(media_fields["summary"].string_value, "done")
    self.assertEqual(media_fields["status"].string_value, "ok")
    self.assertEqual(media_fields["label"].string_value, "done:ok")
    self.assertEqual(media_fields["extra_note"].string_value, "kept")
    self.assertEqual(
        media_fields["image"].content_value.image.data,
        b"\x89PNG",
    )
    self.assertEqual(
        struct_converter.to_json_fallback(media_report),
        {
            "summary": "done",
            "status": "ok",
            "image": {"mime_type": "image/png"},
            "label": "done:ok",
            "extra_note": "kept",
        },
    )

  def test_pydantic_root_model_dict_and_primitive(self):
    class DictRoot(pydantic.RootModel[dict[str, Any]]):
      pass

    class IntRoot(pydantic.RootModel[int]):
      pass

    dict_root = DictRoot(
        {"count": 3, "img": types.Image(data=b"abc", mime_type="image/png")}
    )
    self.assertTrue(struct_converter.is_structured(dict_root))
    dict_fields = {
        f.name: f.value for f in struct_converter.to_struct(dict_root).fields
    }
    self.assertEqual(dict_fields["count"].number_value, 3.0)
    self.assertEqual(dict_fields["img"].content_value.image.data, b"abc")
    self.assertEqual(
        struct_converter.to_json_fallback(dict_root),
        {"count": 3, "img": {"mime_type": "image/png"}},
    )

    plain_dict_root = DictRoot({"count": 5})
    self.assertEqual(
        struct_converter.to_json_fallback(plain_dict_root), {"count": 5}
    )

    int_root = IntRoot(7)
    self.assertFalse(struct_converter.is_structured(int_root))
    int_struct = struct_converter.to_struct(int_root)
    self.assertEqual(int_struct.fields[0].name, "result")
    self.assertEqual(int_struct.fields[0].value.number_value, 7.0)
    self.assertEqual(struct_converter.to_json_fallback(int_root), 7)

  def test_to_json_fallback_inline_media_omits_raw_bytes(self):
    sdk_img = types.Image(data=b"\x89PNG\r\n\x1a\n", mime_type="image/png")
    self.assertEqual(
        struct_converter.to_json_fallback({"screenshot": sdk_img}),
        {"screenshot": {"mime_type": "image/png"}},
    )

    inline_cases = [
        (
            "image",
            content_pb2.ImageContent(
                data=b"\x89PNG\r\n\x1a\n", mime_type_string="image/png"
            ),
            "image/png",
        ),
        (
            "audio",
            content_pb2.AudioContent(
                data=b"RIFF", mime_type_string="audio/wav"
            ),
            "audio/wav",
        ),
        (
            "video",
            content_pb2.VideoContent(
                data=b"ftyp", mime_type_string="video/mp4"
            ),
            "video/mp4",
        ),
        (
            "document",
            content_pb2.DocumentContent(
                data=b"%PDF", mime_type_string="application/pdf"
            ),
            "application/pdf",
        ),
    ]
    for field_name, inline_proto, expected_mime in inline_cases:
      with self.subTest(field_name):
        self.assertEqual(
            struct_converter.to_json_fallback(inline_proto),
            {"mime_type": expected_mime},
        )
        wrapped = content_pb2.Content(**{field_name: inline_proto})
        self.assertEqual(
            struct_converter.to_json_fallback(
                content_pb2.Value(content_value=wrapped)
            ),
            {"mime_type": expected_mime},
        )

    self.assertEqual(
        struct_converter.to_json_fallback(content_pb2.Content()), ""
    )

  def test_format_diff_blocks(self):
    diff_blocks = [
        {
            "start_line": 10,
            "end_line": 12,
            "lines": [
                {"action": "LINE_ACTION_NONE", "text": "def foo():"},
                {"action": "LINE_ACTION_DELETE", "text": "  return 1"},
                {"action": "LINE_ACTION_INSERT", "text": "  return 2"},
            ],
        },
    ]
    self.assertEqual(
        struct_converter.format_diff_blocks(diff_blocks),
        "@@ -10 +12 @@\n def foo():\n-  return 1\n+  return 2",
    )
    self.assertEqual(struct_converter.format_diff_blocks([]), "")


if __name__ == "__main__":
  absltest.main()
