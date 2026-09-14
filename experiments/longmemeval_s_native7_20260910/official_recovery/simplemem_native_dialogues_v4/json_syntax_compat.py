"""Lossless repair of bare consecutive string values for the location field only."""
import json


def repair_locations(text):
    decoder=json.JSONDecoder()
    changes=[]
    i=0
    def whitespace(pos):
        while pos<len(text) and text[pos].isspace(): pos+=1
        return pos
    while i<len(text):
        if text[i]!='"':
            i+=1
            continue
        try:
            token,end=decoder.raw_decode(text,i)
        except json.JSONDecodeError:
            i+=1
            continue
        if not isinstance(token,str):
            i=end
            continue
        after=whitespace(end)
        previous=text[:i].rstrip()[-1:]  # A key must be at an object member boundary.
        if token!='location' or after>=len(text) or text[after]!=':' or previous not in ('{',','):
            i=end
            continue
        start=whitespace(after+1)
        try: first,last=decoder.raw_decode(text,start)
        except json.JSONDecodeError:
            i=end
            continue
        if not isinstance(first,str):
            i=last
            continue
        values=[first]
        stop=last
        while True:
            comma=whitespace(stop)
            if comma>=len(text) or text[comma]!=',': break
            candidate=whitespace(comma+1)
            if candidate>=len(text) or text[candidate]!='"': break
            try: value,value_end=decoder.raw_decode(text,candidate)
            except json.JSONDecodeError: break
            following=whitespace(value_end)
            if following<len(text) and text[following]==':': break  # The next object key.
            if not isinstance(value,str) or following>=len(text) or text[following] not in ',}': break
            values.append(value)
            stop=value_end
        if len(values)>1:
            changes.append({'start':start,'end':stop,'values':values})
        i=stop
    repaired=text
    for item in reversed(changes):
        repaired=repaired[:item['start']]+json.dumps(item['values'],ensure_ascii=False)+repaired[item['end']:]
    return repaired,changes
