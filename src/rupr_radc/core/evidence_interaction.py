import json

def record_spans(row, messages, tokenizer, inputs, prompt, image_token_id=None):
    """Bind every record to character offsets and actual expanded input tokens."""
    ids=inputs['input_ids'][0].tolist()
    decoded=tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    encoded=tokenizer(decoded,add_special_tokens=False,return_offsets_mapping=True)
    assert encoded['input_ids']==ids,'Decoded input does not round-trip to the exact actual token IDs'
    byid={s['source_id']:s for s in row['sources']}
    assert len(byid)==len(row['sources']) and sorted(byid)==sorted(row['order'])
    sources=[byid[k] for k in row['order']]
    payload=json.dumps(dict(news=row['news'],retrieval_records=sources),ensure_ascii=False)
    text=prompt+'\n\n'+payload
    actual_texts=[v['text'] for m in messages for v in m['content'] if v['type']=='text']
    assert actual_texts==[text],'Pooling must use the unchanged complete Reader request'
    # The tokenizer composes Unicode accents (e.g. e + combining acute -> é).
    # Locate spans in the same normalized text as its decoded token sequence;
    # the original model request and its token IDs are never rewritten here.
    normalizer=tokenizer.backend_tokenizer.normalizer
    normalize=normalizer.normalize_str if normalizer is not None else lambda value:value
    # Gemma's SentencePiece normalizer maps spaces to an internal marker; its
    # decoder restores ordinary spaces. Prefer the exact original request when
    # it already occurs in the actual decoded sequence, without changing IDs.
    if text in decoded:
        normalize=lambda value:value
    text=normalize(text)
    start=decoded.index(text);assert decoded.count(text)==1
    base=start+len(normalize(prompt))+2
    offsets=encoded['offset_mapping']
    image_id=tokenizer.convert_tokens_to_ids('<|image_pad|>') if image_token_id is None else int(image_token_id)
    image=[i for i,t in enumerate(ids) if t==image_id];assert image
    ranges=[dict(kind='image',source_id=None,token_indices=image)]
    def add(kind,sid,left,right):
        token_indices=[i for i,(a,b) in enumerate(offsets) if a<right and b>left]
        assert token_indices and all(ids[i]!=image_id for i in token_indices)
        assert min(offsets[i][0] for i in token_indices)<=left and max(offsets[i][1] for i in token_indices)>=right
        ranges.append(dict(kind=kind,source_id=sid,char_start=left,char_end=right,
            text=decoded[left:right],token_indices=token_indices))
    news=normalize(json.dumps(row['news'],ensure_ascii=False));left=base+len('{"news": ')
    assert decoded[left:left+len(news)]==news
    add('news',None,left,left+len(news))
    cursor=left+len(news)
    for source in sources:
        serialized=normalize(json.dumps(source,ensure_ascii=False));pos=decoded.index(serialized,cursor)
        add('evidence',source['source_id'],pos,pos+len(serialized));cursor=pos+len(serialized)
    assert len(ranges)==2+len(sources)
    return ranges
