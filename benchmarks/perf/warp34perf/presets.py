"""Explicit campaign tiers; repetitions are not independent server restarts."""
import copy


def apply_preset(cfg, name):
    if name is None:
        return cfg, None
    preset = cfg.get('presets', {}).get(name)
    if not isinstance(preset, dict):
        raise ValueError(f'campaign does not define preset {name!r}')
    ids = preset.get('cases')
    known = {c['id'] for c in cfg['cases']}
    if (not isinstance(ids, list) or not ids or any(not isinstance(i, str) for i in ids)
            or len(set(ids)) != len(ids) or set(ids) - known):
        raise ValueError('preset cases must be unique existing case IDs')
    repetitions = preset.get('repetitions')
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError('preset repetitions must be a positive integer')
    if name == 'release' and repetitions < 3:
        raise ValueError('release requires at least three repetitions')
    result = copy.deepcopy(cfg)
    result['run']['repetitions'] = repetitions
    result['run']['preset'] = name
    return result, ids
