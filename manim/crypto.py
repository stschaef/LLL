from manimlib.imports import *
import numpy as np

def style_dict_from_string(style):
    if style in [None, ""]: return None
    elements = [s.split(":") for s in style.split(";")]
    elements = [e for e in elements if len(e) == 2]
    dict_elements = {key:value for key, value in elements}

    return dict_elements

def attribute_to_float(attr):
    stripped_attr = "".join([
        char for char in attr
        if char in string.digits + "." + "-"
    ])
    return float(stripped_attr)

def process_val_from_dict(key, D):
    if key not in D:
        return None
    v = D[key]
    
    if v is None:
        return None
    
    if type(v)==str: 
        if v.lower() == "none" or v == "":
            return None
        if v[0] == '#':
            return v.upper()
    
    return attribute_to_float(v)

def process_fill_stroke(element):
    style = style_dict_from_string(element.getAttribute("style"))
    
    if style is None: return None

    fill_color = process_val_from_dict("fill",style)
    opacity    = process_val_from_dict("fill-opacity",style)
    stroke_color = process_val_from_dict("stroke",style)
    stroke_width = process_val_from_dict("stroke-width",style)
    stroke_opacity = process_val_from_dict("stroke-opacity",style)
    
    if fill_color == "NONE" or fill_color=="": fill_color = None
    if stroke_color == "NONE" or stroke_color=="": stroke_color = None
    
    return fill_color, opacity, stroke_color, stroke_width, stroke_opacity

def extract_styles_from_elem(element):
    result = []
    if not isinstance(element, minidom.Element):
        return result
    if element.tagName in ['g', 'svg', 'symbol']:
        result += sum([extract_styles_from_elem(child) for child in element.childNodes],[])
    elif element.tagName in ['circle','rect','ellipse','path','polygon','polyline']:
        result.append(process_fill_stroke(element))
    return [r for r in result if r is not None]

def parse_styles(svg):
    doc = minidom.parse(svg.file_path)
    styles = []
    
    for svg_elem in doc.getElementsByTagName("svg"):
        styles += extract_styles_from_elem(svg_elem)
        
    doc.unlink()
    
    return styles

def color_svg_like_file(svgmobject):
    if not svgmobject.unpack_groups:
        raise Exception("Coloring groups not implemented yet!")
    styles = parse_styles(svgmobject)
    
    for i,(elem,style) in enumerate(zip(svgmobject,styles)):
        fc, alpha, sc, sw, salpha = style
        
        if alpha == 0. or fc is None:
            alpha = 0.
            fc = None
        
        if sw == 0. or sw is None or sc is None or salpha==0.:
            salpha = 0.
            sw = 0.
            sc = None
            
        svgmobject[i].set_fill(color=fc,opacity=alpha)
        svgmobject[i].set_stroke(color=sc,width=sw,opacity=salpha)

class Key(SVGMobject):
    def __init__(self):
        SVGMobject.__init__(self, file_name="key.svg")

class PublicKey(GraphScene):
    def construct(self):
        key = Key()

        snoopy = SVGMobject("snoopy.svg")
        snoopy.set_color(BLUE_C)
        snoopy.scale(1.5)
        snoopy.to_corner(UL)
        snoopy.shift(LEFT*.2)

        woodstock = SVGMobject("Woodstock.svg")
        woodstock.to_corner(UR)
        woodstock.set_color(YELLOW_C)
        woodstock.scale(1.5)
        woodstock.shift(RIGHT*2)
        woodstock.shift(DOWN*.25)
        self.play(Write(snoopy), Write(woodstock))
        self.wait(1)

        ntru = TexMobject(r"\textbf{NTRU}")
        ntru.scale(2)

        self.play(Write(ntru))
        self.wait(2)
        self.play(FadeOut(ntru))

        public = TextMobject(r"\underline{Public}")
        self.play(Write(public))
        self.wait(2)

        params = TexMobject("(", "N", ",", r"p", ",", r"q", ")")
        params.next_to(public, direction=DOWN)

        conditions = TexMobject("N", r"\text{ prime, }", "p", "<", "q", r",\ ", r"\text{gcd}(", "p", ",","q", ") = 1")
    
        conditions.to_edge(UP)
        self.play(Write(params))
        self.play(Write(conditions))
        self.wait(2)
        self.play(FadeOutAndShift(conditions))

        private = TextMobject(r"\underline{Private}")
        private.to_edge(RIGHT)
        self.play(Write(private))

        polys = TexMobject("f", ",", "g")
        polys.set_color_by_tex("f", BLUE)
        polys.set_color_by_tex("g", PINK)
        poly_conditions = TexMobject("f", ",", "g", r"\in R[X]/(X^N - 1)")
        poly_conditions.set_color_by_tex("f", BLUE)
        poly_conditions.set_color_by_tex("g", PINK)
        poly_conditions.set_color_by_tex(r"\text", WHITE)

        polys.next_to(private, direction=DOWN)
        poly_conditions.to_edge(UP)

        coeffs_cond = TexMobject(r"\text{ random, coeffiecients in }", r"\{-1,0,1\}")
        coeffs_cond.next_to(poly_conditions, direction=DOWN)
        coeffs_cond.shift(RIGHT*.15)

        invertible = TexMobject("f", r"\text{ invertible }", r"\text{mod } p, \pmod q")
        invertible.set_color_by_tex("f", BLUE)
        invertible.next_to(coeffs_cond, direction=DOWN)


        self.play(Write(polys))
        self.wait(2)
        self.play(Write(poly_conditions), Write(coeffs_cond), Write(invertible))
        self.wait(3)

        private_key = Key()
        private_key.set_color_by_gradient(BLUE, GREEN)
        private_key.next_to(polys, direction=LEFT)
        self.play(Write(private_key))

        self.play(*[FadeOutAndShift(o) for o in [poly_conditions, coeffs_cond, invertible]])
        self.wait(2)

        h = TexMobject("h", r"\equiv", r"p \cdot", r"f_p", "g", r" \pmod  q")
        h.set_color_by_tex("f", BLUE)
        h.set_color_by_tex("g", PINK)
        h.set_color_by_tex("h", ORANGE)

        h.to_edge(UP)
        self.play(Write(h))
        self.wait(2)

        h_pub = TexMobject(r"\text{Public key }", "h")
        h_pub[1].set_color(ORANGE)
        h_pub.next_to(params, direction=DOWN)

        pub_key = Key()
        pub_key.set_color_by_gradient(BLUE, GREEN)
        pub_key.next_to(h_pub, direction=LEFT)
        self.play(Write(pub_key))

        self.play(Transform(h, h_pub), Write(pub_key))
        self.wait(2)

        msg = TexMobject("m", r"\text{ message as a polynomial}")
        msg[0].set_color(GREEN)
        msg_conds = TextMobject("coefficients in $[-p/2, p/2]$")
        rand_r = TexMobject(r"\text{random polynomial", "r")

        msg.to_edge(UP)
        msg_conds.next_to(msg, direction=DOWN)
        rand_r.next_to(msg_conds, direction=DOWN)

        private_snoop = TextMobject(r"\underline{Private}")
        private_snoop.to_edge(LEFT)
        self.play(Write(private_snoop))

        msg_snoop = TexMobject(r"\text{Message }", "m")
        msg_snoop[1].set_color(GREEN)
        msg_snoop.next_to(private_snoop, direction=DOWN)
        msg_snoop.to_edge(LEFT)

        r_snoop = TexMobject(r"\text{Random }", "r")
        r_snoop.next_to(msg_snoop, direction=DOWN)
        r_snoop.to_edge(LEFT)

        self.play(*[Write(o) for o in [msg, msg_conds, rand_r]])
        self.wait(2)

        self.play(FadeOutAndShift(msg_conds), Transform(rand_r, r_snoop), Transform(msg, msg_snoop))

        e = TexMobject("e", r"\equiv", "r", "h", "+", "m", r" \pmod q")
        e[0].set_color(GREEN)
        e[5].set_color(GREEN)
        e[3].set_color(ORANGE)

        e.to_edge(UP)
        self.play(Write(e))
        self.wait(2)

        e_snoop = TexMobject(r"\text{Encrypted }", "e")
        e_snoop[1].set_color(GREEN)
        e_snoop.next_to(rand_r, direction=DOWN)
        e_snoop.to_edge(LEFT)

        self.play(Transform(e, e_snoop))
        self.wait(2)

        msg_e = TexMobject("e")
        msg_e.set_color(GREEN)
        msg_e.scale(2)
        msg_e.next_to(snoopy, buff=1)
        orig_y = msg_e.get_y()


        mail = SVGMobject("mail.svg")
        mail.scale(.5)
        mail.set_color(GREEN)
        mail.next_to(msg_e, direction=DOWN)

        mail.add_updater(lambda obj: obj.next_to(msg_e, direction=DOWN))

        
        self.play(Write(mail), ReplacementTransform(copy.deepcopy(e_snoop), msg_e))
        self.wait(2)

        msg_e.generate_target()
        msg_e.target.next_to(woodstock, direction=LEFT, buff=1)
        msg_e.target.set_y(orig_y)

        self.play(MoveToTarget(msg_e))

        self.wait(2)

        a_eq = TexMobject(r"a \equiv ", "f", "e", r" \pmod q")
        a_eq[1].set_color(BLUE)
        a_eq[2].set_color(GREEN) 
        a_eq.to_edge(UP)

        b_eq = TexMobject(r"b \equiv a \pmod p \equiv f \cdot ", "m", r" \pmod p")
        b_eq[1].set_color(GREEN)
        b_eq.next_to(a_eq, direction=DOWN)


        c_eq = TexMobject(r"c \equiv ", "f_p", r" \cdot b \equiv ", "m", r" \pmod p")
        c_eq[1].set_color(BLUE)
        c_eq[3].set_color(GREEN)
        c_eq.next_to(b_eq, direction=DOWN)

        self.play(FadeOut(msg_e), FadeOut(mail))
        self.play(Write(a_eq))
        self.wait(2)
        self.play(Write(b_eq))
        self.wait(2)
        self.play(Write(c_eq))
        self.wait(2)


        
        