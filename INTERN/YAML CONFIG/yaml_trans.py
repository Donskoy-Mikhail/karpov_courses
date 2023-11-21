import yaml
import re


def yaml_to_env(config_text: str) -> str:
    config = yaml.load(config_text, Loader=yaml.FullLoader)
    env_vars = []
    _flatten_dict(config, '', env_vars)
    return '\n'.join(env_vars)


def env_to_yaml(env_list: str) -> str:
    config = {}
    for line in env_list.split('\n'):
        line = line.strip()
        if not line:
            continue
        key, value = line.split('=', 1)

        # Пробуем преобразовать значение в число или булево значение
        try:
            if '.' in value:
                value = float(value)
            else:
                int_value = int(value)
                if str(int_value) == value:  # Убедимся, что это не было числом с плавающей точкой, преобразованным в int
                    value = int_value
            if value == 'True':
                value = True
            elif value == 'False':
                value = False
        except ValueError:
            pass

        # Разделите ключ на части и создайте вложенные словари
        keys = key.split('.')
        current = config
        for k in keys[:-1]:
            current = current.setdefault(k, {})
        current[keys[-1]] = value

    return re.sub("'False'", 'false',
                  re.sub("'True'", 'true',
                         yaml.dump(config, default_flow_style=False, default_style='', allow_unicode=True)
                         , count=0, flags=0))


def _flatten_dict(d, parent_key, env_vars):
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            _flatten_dict(v, new_key, env_vars)
        else:
            env_vars.append(f"{new_key}={v}")
