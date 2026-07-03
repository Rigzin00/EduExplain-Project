import sys; sys.path.insert(0, '.')
from src.extraction.maha_parser import MahaParser
from collections import Counter
import json

mp = MahaParser(subject='Biology', source='MAHA', chapter_no=1, chapter_title='')
chunks = mp.parse(
    'Maharashtra/BIO/class11_biology_chapters/chapter_01/Bio11_Ch1.json',
    'output/bio11_ch01_test.json'
)

print(f'Chapter title : {mp.chapter_title!r}')
print(f'Total chunks  : {len(chunks)}')
print()
print('Chunk type breakdown:')
for t, n in Counter(c["chunk_type"] for c in chunks).most_common():
    print(f'  {t:<20} {n}')
print()
print('figure_references hits:')
hit = False
for c in chunks:
    if c.get('figure_references'):
        print(f'  [{c["chunk_type"]}] {c["figure_references"]}')
        hit = True
if not hit:
    print('  (none found - no Fig. references in text)')
print()
print('First 4 chunks (type + text preview):')
for c in chunks[:4]:
    print(f'  [{c["chunk_type"]}] section={c["section"]!r} | {c["text"][:90]!r}')
