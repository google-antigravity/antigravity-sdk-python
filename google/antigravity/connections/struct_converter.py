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

"""Structured data conversion utilities for SDK connections and genai.Content."""

from collections import abc
import dataclasses
import enum
from typing import Any, cast

from google.protobuf import message
import pydantic

from google.antigravity.proto import content_pb2
from google.antigravity import types

# SDK media primitives inherit pydantic.BaseModel, so _as_mapping treats them
# as leaf values serialized into Content protos rather than generic Structs.
_SDK_MEDIA_TYPES = (types.Image, types.Audio, types.Video, types.Document)

# Automatically discover all Content and media component Message classes from
# content.proto:
_CONTENT_PROTO_TYPES: tuple[type[Any], ...] = tuple(
    v
    for v in content_pb2.__dict__.values()
    if isinstance(v, type)
    and issubclass(v, message.Message)
    and getattr(v, "DESCRIPTOR", None) is not None
    and v.DESCRIPTOR.file.name.endswith("content.proto")
    and (
        v.DESCRIPTOR.name.endswith("Content")
        or v.DESCRIPTOR.name.endswith("Options")
    )
)

_CONTENT_DESCRIPTOR_NAMES: frozenset[str] = frozenset(
    cls.DESCRIPTOR.name for cls in _CONTENT_PROTO_TYPES
)

_MEDIA_TYPE_MAP = {
    content_pb2.VideoContent: "video",
    content_pb2.AudioContent: "audio",
    content_pb2.ImageContent: "image",
    content_pb2.DocumentContent: "document",
    content_pb2.TextContent: "text",
}

_MEDIA_DESCRIPTOR_MAP = {
    "VideoContent": "video",
    "AudioContent": "audio",
    "ImageContent": "image",
    "DocumentContent": "document",
    "TextContent": "text",
}


def is_content_proto(obj: Any) -> bool:
  """Returns True if obj is an interactions Content proto or media component."""
  if isinstance(obj, _CONTENT_PROTO_TYPES):
    return True
  if (
      hasattr(obj, "DESCRIPTOR")
      and obj.DESCRIPTOR.name in _CONTENT_DESCRIPTOR_NAMES
  ):
    return True
  return False


def _has_raw_leaf_object(obj: Any) -> bool:
  """Returns True if `obj` contains SDK media or Content proto instances."""
  if is_content_proto(obj) or isinstance(obj, _SDK_MEDIA_TYPES):
    return True
  if isinstance(obj, pydantic.RootModel):
    return _has_raw_leaf_object(obj.root)
  if isinstance(obj, pydantic.BaseModel):
    model_cls = type(obj)
    for name, info in model_cls.model_fields.items():
      if not info.exclude and _has_raw_leaf_object(getattr(obj, name)):
        return True
    for name, info in model_cls.model_computed_fields.items():
      if not getattr(info, "exclude", False) and _has_raw_leaf_object(
          getattr(obj, name)
      ):
        return True
    extra = getattr(obj, "__pydantic_extra__", None) or {}
    return any(_has_raw_leaf_object(v) for v in extra.values())
  if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
    return any(
        _has_raw_leaf_object(getattr(obj, f.name))
        for f in dataclasses.fields(obj)
    )
  if isinstance(obj, abc.Mapping):
    return any(_has_raw_leaf_object(v) for v in obj.values())
  if isinstance(obj, abc.Sequence) and not isinstance(obj, (str, bytes)):
    return any(_has_raw_leaf_object(v) for v in obj)
  return False


def _as_mapping(obj: Any) -> abc.Mapping[str, Any] | None:
  """Converts structured objects to a Mapping, or None for primitives and media."""
  if (
      is_content_proto(obj)
      or isinstance(obj, _SDK_MEDIA_TYPES)
      or isinstance(obj, type)
  ):
    return None
  if isinstance(obj, abc.Mapping):
    return obj
  if isinstance(obj, pydantic.RootModel):
    if _has_raw_leaf_object(obj.root):
      return _as_mapping(obj.root)
    try:
      d = obj.model_dump()
      return d if isinstance(d, abc.Mapping) else None
    except (TypeError, ValueError, AttributeError):
      return None
  # Preserve nested SDK media and Content proto leaves on BaseModel instances
  # without bypassing model_dump() serialization rules for plain models.
  if isinstance(obj, pydantic.BaseModel):
    try:
      if _has_raw_leaf_object(obj):
        model_cls = type(obj)
        out: dict[str, Any] = {
            k: getattr(obj, k)
            for k, info in model_cls.model_fields.items()
            if not info.exclude
        }
        for k, info in model_cls.model_computed_fields.items():
          if not getattr(info, "exclude", False):
            out[k] = getattr(obj, k)
        if obj.__pydantic_extra__:
          out.update(obj.__pydantic_extra__)
        return out
      d = obj.model_dump()
      if isinstance(d, abc.Mapping):
        return d
    except (TypeError, ValueError, AttributeError):
      pass
  if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
    try:
      d = obj.model_dump()
      if isinstance(d, abc.Mapping):
        return d
    except (TypeError, ValueError, AttributeError):
      pass
  if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
    try:
      d = obj.dict()
      if isinstance(d, abc.Mapping):
        return d
    except (TypeError, ValueError, AttributeError):
      pass
  if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
    try:
      return {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
    except (TypeError, ValueError, AttributeError):
      pass
  return None


def is_structured(obj: Any) -> bool:
  """Returns True if obj is a mapping or non-media Pydantic/dataclass instance."""
  return _as_mapping(obj) is not None


def to_struct(obj: Any) -> content_pb2.Struct:
  """Converts a mapping, Pydantic model, or dataclass into a content_pb2.Struct protobuf message.

  Args:
    obj: The Python mapping, Pydantic model, or dataclass to serialize.

  Returns:
    A populated content_pb2.Struct protobuf message.
  """
  if isinstance(obj, content_pb2.Struct) or (
      hasattr(obj, "DESCRIPTOR") and obj.DESCRIPTOR.name == "Struct"
  ):
    return obj

  m = _as_mapping(obj)
  if m is not None:
    fields = [
        content_pb2.Field(name=str(k), value=value_to_proto(v))
        for k, v in m.items()
    ]
    return content_pb2.Struct(fields=fields)

  return content_pb2.Struct(
      fields=[content_pb2.Field(name="result", value=value_to_proto(obj))]
  )


def dict_to_struct(d: abc.Mapping[str, Any]) -> content_pb2.Struct:
  """Converts a Python mapping into a content_pb2.Struct protobuf message.

  Args:
    d: The Python mapping to serialize.

  Returns:
    A populated content_pb2.Struct protobuf message.
  """
  return to_struct(d)


def _wrap_media_content(field_name: str, val: Any) -> content_pb2.Content:
  """Constructs a Content message with the given field set."""
  content_cls = cast(Any, content_pb2.Content)
  return content_cls(**{field_name: val})


def _sdk_media_to_content_proto(obj: Any) -> content_pb2.Content | None:
  """Converts an SDK media object into a content_pb2.Content message."""
  if isinstance(obj, types.Image):
    return content_pb2.Content(
        image=content_pb2.ImageContent(
            data=obj.data, mime_type_string=obj.mime_type
        )
    )
  if isinstance(obj, types.Audio):
    return content_pb2.Content(
        audio=content_pb2.AudioContent(
            data=obj.data, mime_type_string=obj.mime_type
        )
    )
  if isinstance(obj, types.Video):
    return content_pb2.Content(
        video=content_pb2.VideoContent(
            data=obj.data, mime_type_string=obj.mime_type
        )
    )
  if isinstance(obj, types.Document):
    return content_pb2.Content(
        document=content_pb2.DocumentContent(
            data=obj.data, mime_type_string=obj.mime_type
        )
    )
  return None


def value_to_proto(val: Any) -> content_pb2.Value:
  """Converts a Python value or SDK media object into a content_pb2.Value proto.

  Args:
    val: The Python value, SDK media object, or data structure to serialize.

  Returns:
    A populated content_pb2.Value protobuf message.
  """
  if isinstance(val, enum.Enum):
    val = val.value
  if isinstance(val, pydantic.RootModel):
    val = val.root
  if val is None:
    return content_pb2.Value(null_value="NULL_VALUE")
  if isinstance(val, bool):
    return content_pb2.Value(bool_value=val)
  if isinstance(val, (int, float)):
    return content_pb2.Value(number_value=float(val))
  if isinstance(val, str):
    return content_pb2.Value(string_value=val)
  if isinstance(val, abc.Sequence) and not isinstance(val, (str, bytes)):
    return content_pb2.Value(
        list_value=content_pb2.ListValue(
            values=[value_to_proto(x) for x in val]
        )
    )

  sdk_content = _sdk_media_to_content_proto(val)
  if sdk_content is not None:
    return content_pb2.Value(content_value=sdk_content)

  m = _as_mapping(val)
  if m is not None:
    return content_pb2.Value(struct_value=to_struct(m))

  if isinstance(val, content_pb2.Value):
    return val
  if isinstance(val, content_pb2.Struct):
    return content_pb2.Value(struct_value=val)
  if isinstance(val, content_pb2.ListValue):
    return content_pb2.Value(list_value=val)
  if isinstance(val, content_pb2.Content):
    return content_pb2.Value(content_value=val)

  for media_type, field_name in _MEDIA_TYPE_MAP.items():
    if isinstance(val, media_type):
      return content_pb2.Value(
          content_value=_wrap_media_content(field_name, val)
      )

  if hasattr(val, "DESCRIPTOR"):
    val_any = cast(Any, val)
    desc_name = val_any.DESCRIPTOR.name
    if desc_name == "Value":
      return val_any
    elif desc_name == "Struct":
      return content_pb2.Value(struct_value=val_any)
    elif desc_name == "ListValue":
      return content_pb2.Value(list_value=val_any)
    elif desc_name == "Content":
      return content_pb2.Value(content_value=val_any)
    elif desc_name in _MEDIA_DESCRIPTOR_MAP:
      return content_pb2.Value(
          content_value=_wrap_media_content(
              _MEDIA_DESCRIPTOR_MAP[desc_name], val_any
          )
      )

  return content_pb2.Value(string_value=str(val))


def _get_display_name(obj: Any) -> str:
  """Extracts display_name from a Content or media component object if available."""
  try:
    if hasattr(obj, "display_name") and obj.display_name:
      return obj.display_name
  except (AttributeError, ValueError):
    pass
  if hasattr(obj, "WhichOneof"):
    try:
      oneof_field = obj.WhichOneof("type")
      if oneof_field:
        inner = getattr(obj, oneof_field)
        if hasattr(inner, "display_name") and inner.display_name:
          return inner.display_name
    except (AttributeError, ValueError):
      pass
  return ""


def _get_uri(obj: Any) -> str:
  """Extracts uri from a Content or media component object if available."""
  try:
    if hasattr(obj, "uri") and obj.uri:
      return obj.uri
  except (AttributeError, ValueError):
    pass
  if hasattr(obj, "WhichOneof"):
    try:
      oneof_field = obj.WhichOneof("type")
      if oneof_field:
        inner = getattr(obj, oneof_field)
        if hasattr(inner, "uri") and inner.uri:
          return inner.uri
    except (AttributeError, ValueError):
      pass
  return ""


def to_json_fallback(obj: Any) -> Any:
  """Converts objects into JSON-serializable primitives for fallback.

  Args:
    obj: The Python object or data structure to convert.

  Returns:
    A JSON-serializable representation, replacing URI or named Content instances
    with `$ref` pointers and inline binary media with `{"mime_type": ...}`
    summaries that omit raw bytes.
  """
  if isinstance(obj, enum.Enum):
    obj = obj.value
  if isinstance(obj, pydantic.RootModel):
    obj = obj.root
  if isinstance(obj, _SDK_MEDIA_TYPES):
    return {"mime_type": obj.mime_type}

  if is_content_proto(obj):
    dn = _get_display_name(obj)
    if dn:
      return {"$ref": dn}
    uri = _get_uri(obj)
    if uri:
      return {"$ref": uri}
    if isinstance(obj, content_pb2.TextContent):
      return obj.text
    if isinstance(obj, content_pb2.Content):
      which = obj.WhichOneof("type")
      if which == "text":
        return obj.text.text
      if which in ("image", "audio", "video", "document"):
        return {"mime_type": getattr(obj, which).mime_type_string}
    for media_type, field_name in _MEDIA_TYPE_MAP.items():
      if field_name != "text" and isinstance(obj, media_type):
        return {"mime_type": getattr(obj, "mime_type_string", "")}
    return str(obj)

  if isinstance(obj, content_pb2.Struct) or (
      hasattr(obj, "DESCRIPTOR") and obj.DESCRIPTOR.name == "Struct"
  ):
    return {f.name: to_json_fallback(f.value) for f in obj.fields}

  if isinstance(obj, content_pb2.ListValue) or (
      hasattr(obj, "DESCRIPTOR") and obj.DESCRIPTOR.name == "ListValue"
  ):
    return [to_json_fallback(v) for v in obj.values]

  if isinstance(obj, content_pb2.Value) or (
      hasattr(obj, "DESCRIPTOR") and obj.DESCRIPTOR.name == "Value"
  ):
    which = obj.WhichOneof("kind")
    if which == "null_value":
      return None
    elif which == "bool_value":
      return obj.bool_value
    elif which == "number_value":
      return obj.number_value
    elif which == "string_value":
      return obj.string_value
    elif which == "struct_value":
      return to_json_fallback(obj.struct_value)
    elif which == "list_value":
      return to_json_fallback(obj.list_value)
    elif which == "content_value":
      return to_json_fallback(obj.content_value)
    return str(obj)

  m = _as_mapping(obj)
  if m is not None:
    return {k: to_json_fallback(v) for k, v in m.items()}

  if isinstance(obj, abc.Sequence) and not isinstance(obj, (str, bytes)):
    return [to_json_fallback(v) for v in obj]

  if isinstance(obj, (str, int, float, bool, type(None))):
    return obj

  return str(obj)


def unwrap_wire_value(val: Any) -> Any:
  """Recursively unwraps MessageToDict-serialized genai.Value dict into native Python values.

  `genai.Struct` and `genai.Value` from interactions content.proto use fields
  like `fields = 1`, `string_value = 1`, `struct_value = 5`, etc.
  When serialized with `json_format.MessageToDict`, values appear as
  `{"string_value": "..."}` or `{"stringValue": "..."}`, etc. This helper
  recursively extracts the inner native Python values.

  Args:
    val: The serialized `genai.Value` dict or primitive value to unwrap.

  Returns:
    The unwrapped native Python value.
  """
  if not isinstance(val, dict):
    return val
  for key in ("string_value", "stringValue"):
    if key in val:
      return val[key]
  for key in ("number_value", "numberValue"):
    if key in val:
      return val[key]
  for key in ("bool_value", "boolValue"):
    if key in val:
      return val[key]
  for key in ("null_value", "nullValue"):
    if key in val:
      return None
  for key in ("struct_value", "structValue"):
    if key in val and isinstance(val[key], dict):
      return unwrap_wire_struct(val[key])
  for key in ("list_value", "listValue"):
    if key in val and isinstance(val[key], dict):
      values = val[key].get("values", [])
      if isinstance(values, list):
        return [unwrap_wire_value(v) for v in values]
  for key in ("content_value", "contentValue"):
    if key in val:
      return val[key]
  return val


def unwrap_wire_struct(val: Any) -> dict[str, Any]:
  """Unpacks a MessageToDict-serialized genai.Struct dict ({'fields': [...]}) into a Python dict.

  If val is already a plain dict without a 'fields' list, it is returned as is.
  If val is not a dict, an empty dict is returned.

  Args:
    val: The serialized `genai.Struct` dict or plain mapping to unwrap.

  Returns:
    The unpacked Python dictionary.
  """
  if not isinstance(val, dict):
    return {}
  if "fields" in val and isinstance(val["fields"], list):
    unpacked = {}
    for item in val["fields"]:
      if isinstance(item, dict) and "name" in item:
        unpacked[item["name"]] = unwrap_wire_value(item.get("value"))
    return unpacked
  return val


_DIFF_ACTION_PREFIX = {
    "LINE_ACTION_INSERT": "+",
    "LINE_ACTION_DELETE": "-",
}


def format_diff_blocks(
    diff_blocks: abc.Sequence[abc.Mapping[str, Any]],
) -> str:
  """Formats `EditFileResultStep.diff_block` dicts as a unified diff string.

  Args:
    diff_blocks: Sequence of serialized `DiffBlock` mappings (`start_line`,
      `end_line`, and `lines` with `action` and `text`).

  Returns:
    A human-readable unified diff string.
  """
  chunks: list[str] = []
  for block in diff_blocks:
    if not isinstance(block, abc.Mapping):
      continue
    start = block.get("start_line", 0)
    end = block.get("end_line", 0)
    lines = [f"@@ -{start} +{end} @@"]
    for line in block.get("lines") or ():
      if not isinstance(line, abc.Mapping):
        continue
      prefix = _DIFF_ACTION_PREFIX.get(str(line.get("action", "")), " ")
      lines.append(f"{prefix}{line.get('text', '')}")
    chunks.append("\n".join(lines))
  return "\n".join(chunks)
