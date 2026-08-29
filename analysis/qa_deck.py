"""Geometry QA: flag shapes outside the canvas and estimate text overflow."""
from pptx import Presentation
from pptx.util import Emu
import sys
P=Presentation('deck/GHI_Forecasting_Defence.pptx')
W=P.slide_width/914400; H=P.slide_height/914400
def est_lines(text, width_in, size_pt):
    # ~1.85 chars per pt of width at Calibri
    cpl = max(1, int(width_in*72/(size_pt*0.50)))
    n=0
    for para in text.split('\n'):
        n += max(1, -(-len(para)//cpl))
    return n
issues=[]
for i,s in enumerate(P.slides,1):
    for sh in s.shapes:
        l,t = sh.left/914400, sh.top/914400
        w,h = (sh.width or 0)/914400, (sh.height or 0)/914400
        if l < -0.02 or t < -0.02 or l+w > W+0.02 or t+h > H+0.02:
            issues.append(f'slide {i}: OFF-CANVAS {sh.shape_type} "{(sh.text_frame.text[:40] if sh.has_text_frame else "")}" '
                          f'box=({l:.2f},{t:.2f},{w:.2f},{h:.2f})')
        if sh.has_text_frame and sh.text_frame.text.strip():
            tf=sh.text_frame
            tot=0
            for p in tf.paragraphs:
                txt=''.join(r.text for r in p.runs)
                if not txt: continue
                sz=max([ (r.font.size.pt if r.font.size else 12) for r in p.runs] or [12])
                tot += est_lines(txt, w, sz)*sz*1.22/72
            if tot > h + 0.30 and h>0:
                issues.append(f'slide {i}: OVERFLOW? est {tot:.2f}in vs box {h:.2f}in  '
                              f'"{tf.text[:60]!r}"')
print(f'canvas {W}x{H}, slides {len(P.slides._sldIdLst)}')
for x in issues: print(' -',x)
print(f'{len(issues)} issues')
