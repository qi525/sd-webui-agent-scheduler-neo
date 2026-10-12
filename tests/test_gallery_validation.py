import importlib.util
import sys
import types
from pathlib import Path


def install_stubs():
    gradio = types.ModuleType('gradio')
    gradio.Button = type('Button', (), {})
    gradio.Blocks = type('Blocks', (), {})
    gradio.Gallery = type('Gallery', (), {})
    gradio.Textbox = type('Textbox', (), {})
    gradio.TextArea = type('TextArea', (), {})
    gradio.Row = type('Row', (), {})
    gradio.Checkbox = type('Checkbox', (), {})
    gradio.Dropdown = type('Dropdown', (), {})
    gradio.File = type('File', (), {})
    gradio.HTML = type('HTML', (), {})
    gradio.Group = type('Group', (), {})
    gradio.State = type('State', (), {})
    gradio.update = lambda *args, **kwargs: {'update': True}
    sys.modules['gradio'] = gradio

    ns = types.ModuleType('modules')
    ns.call_queue = types.SimpleNamespace(wrap_gradio_call=lambda fn: fn)
    ns.shared = types.ModuleType('modules.shared')
    ns.shared.opts = types.SimpleNamespace()
    ns.shared.list_checkpoint_tiles = lambda: []
    ns.shared.refresh_checkpoints = lambda *args, **kwargs: None
    ns.script_callbacks = types.SimpleNamespace(
        on_ui_tabs=lambda *args, **kwargs: None,
        on_ui_settings=lambda *args, **kwargs: None,
        on_app_started=lambda *args, **kwargs: None,
    )
    ns.scripts = types.ModuleType('modules.scripts')
    ns.scripts.Script = type('Script', (), {})
    ns.scripts.AlwaysVisible = True
    ns.ui_components = types.ModuleType('modules.ui_components')
    ns.ui_components.ToolButton = type('ToolButton', (), {})
    sys.modules['modules'] = ns
    sys.modules['modules.shared'] = ns.shared
    sys.modules['modules.scripts'] = ns.scripts
    sys.modules['modules.ui_components'] = ns.ui_components

    sys.modules['modules.cmd_args'] = types.SimpleNamespace(parser=types.SimpleNamespace(description=''))
    sys.modules['modules.ui'] = types.SimpleNamespace(create_refresh_button=lambda *args, **kwargs: None)
    sys.modules['modules.ui_common'] = types.SimpleNamespace(save_files=lambda *args, **kwargs: None)
    sys.modules['modules.sd_models'] = types.SimpleNamespace(model_path='', select_checkpoint=lambda *args, **kwargs: None)
    sys.modules['modules.infotext_utils'] = types.SimpleNamespace(
        registered_param_bindings=[],
        register_paste_params_button=lambda *args, **kwargs: None,
        connect_paste_params_buttons=lambda *args, **kwargs: None,
        parse_generation_parameters=lambda *args, **kwargs: {},
        ParamBinding=type('ParamBinding', (), {}),
    )

    sys.modules['agent_scheduler.task_runner'] = types.SimpleNamespace(
        TaskRunner=type('TaskRunner', (), {}),
        get_instance=lambda: None,
    )
    sys.modules['agent_scheduler.helpers'] = types.SimpleNamespace(
        get_default_config_dependencies=lambda *args, **kwargs: [],
        log=types.SimpleNamespace(error=lambda *args, **kwargs: None, debug=lambda *args, **kwargs: None, info=lambda *args, **kwargs: None),
        compare_components_with_ids=lambda *args, **kwargs: None,
        get_components_by_ids=lambda *args, **kwargs: [],
        is_macos=False,
    )
    sys.modules['agent_scheduler.db'] = types.SimpleNamespace(
        init=lambda *args, **kwargs: None,
        task_manager=types.SimpleNamespace(get_task=lambda *args, **kwargs: None, delete_tasks=lambda *args, **kwargs: 0),
        TaskStatus=types.SimpleNamespace(DONE='done', FAILED='failed', INTERRUPTED='interrupted'),
    )
    sys.modules['agent_scheduler.api'] = types.SimpleNamespace(regsiter_apis=lambda *args, **kwargs: None)


def load_task_scheduler_module():
    install_stubs()
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'task_scheduler.py'
    spec = importlib.util.spec_from_file_location('task_scheduler_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rejects_false_and_none():
    mod = load_task_scheduler_module()
    assert mod.coerce_gallery_payload(False) == []
    assert mod.coerce_gallery_payload(None) == []


def test_accepts_list_and_tuple():
    mod = load_task_scheduler_module()
    assert mod.coerce_gallery_payload([1, 2, 3]) == [1, 2, 3]
    assert mod.coerce_gallery_payload(("a", "b")) == ["a", "b"]


def test_invalid_index_candidates_are_blocked():
    mod = load_task_scheduler_module()
    for value in [None, False, "", "False", "false", "None", "null", -1, "abc"]:
        assert mod.normalize_gallery_image_idx(value) is None


def test_valid_index_candidates_are_kept():
    mod = load_task_scheduler_module()
    for value in [0, 1, "0", "1", 9, "9"]:
        assert mod.normalize_gallery_image_idx(value) == int(value)
