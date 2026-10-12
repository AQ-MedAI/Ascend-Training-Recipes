"""Select full-table row chunks for one rollout table shard before SHM packing."""


def select_engram_rows(expected, tables):
    selected = dict(expected)
    requests = {}
    for name, shape in expected.items():
        if not name.startswith('engram/') or '.embed.weight.rows.' not in name:
            continue
        layer, interval = name[7:].split('.embed.weight.rows.')
        start, end = map(int, interval.split('.'))
        table = tables[int(layer)]
        if list(shape) != [end - start, table.width] or start < 0 or end <= start:
            raise ValueError('Invalid Engram chunk: ' + name)
        lo, hi = max(start, table.start), min(end, table.end)
        del selected[name]
        if hi <= lo:
            requests[name] = [0, 0, name]
            continue
        target = f'engram/{layer}.embed.weight.rows.{lo}.{hi}'
        if target in selected:
            raise ValueError('Duplicate Engram destination chunk: ' + target)
        selected[target] = [hi - lo, table.width]
        requests[name] = [lo - start, hi - start, target]
    return selected, requests
