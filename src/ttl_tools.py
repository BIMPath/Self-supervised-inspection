"""
ttl_tools.py — a self-contained Turtle parser, serialiser and linter for the CIEO project.

Why this exists
---------------
The reference toolchain for this repository is rdflib + pySHACL + Apache Jena. This module is not
a replacement for any of them. It exists so that the ontology, the shapes and the instance data can
be syntax-checked and cross-referenced in environments where those packages cannot be installed,
and so that the repository has a dependency-free way to prove that every file in it parses and that
no module refers to a term no module defines.

It implements the subset of Turtle (W3C Recommendation, 2014) that CIEO actually uses:
prefix and base directives in both Turtle and SPARQL syntax, prefixed and absolute IRIs, blank node
labels and anonymous blank nodes, collections, the `a` keyword, predicate-object and object lists,
all four string literal forms with language tags and datatypes, and numeric and boolean literals.

It does NOT implement: RDF-star, named graphs, or IRI resolution against a network. Relative IRIs
are resolved only syntactically against @base.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

# ---------------------------------------------------------------------------
# Terms
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IRI:
    value: str

    def __repr__(self) -> str:
        return f"<{self.value}>"


@dataclass(frozen=True)
class BNode:
    label: str

    def __repr__(self) -> str:
        return f"_:{self.label}"


@dataclass(frozen=True)
class Literal:
    value: str
    datatype: str | None = None
    language: str | None = None

    def __repr__(self) -> str:
        if self.language:
            return f'"{self.value}"@{self.language}'
        if self.datatype:
            return f'"{self.value}"^^<{self.datatype}>'
        return f'"{self.value}"'


Term = IRI | BNode | Literal
Triple = tuple[Term, Term, Term]

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDF_FIRST = "http://www.w3.org/1999/02/22-rdf-syntax-ns#first"
RDF_REST = "http://www.w3.org/1999/02/22-rdf-syntax-ns#rest"
RDF_NIL = "http://www.w3.org/1999/02/22-rdf-syntax-ns#nil"
XSD = "http://www.w3.org/2001/XMLSchema#"


class TurtleSyntaxError(Exception):
    def __init__(self, message: str, line: int, col: int, source: str = "<string>"):
        super().__init__(f"{source}:{line}:{col}: {message}")
        self.message = message
        self.line = line
        self.col = col
        self.source = source


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

# Characters permitted in the local part of a prefixed name, per the Turtle grammar's
# PN_LOCAL production, restricted to what does not need percent- or backslash-escaping.
_PN_CHARS_BASE = r"A-Za-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02FF\u0370-\u037D\u037F-\u1FFF\u200C-\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD"
_PN_CHARS_U = _PN_CHARS_BASE + "_"
_PN_CHARS = _PN_CHARS_U + r"\-0-9\u00B7\u0300-\u036F\u203F-\u2040"

_RE_PNAME = re.compile(
    rf"(?P<prefix>[{_PN_CHARS_BASE}](?:[{_PN_CHARS}.]*[{_PN_CHARS}])?)?:"
    rf"(?P<local>(?:[{_PN_CHARS_U}0-9:]|\\[-_~.!$&'()*+,;=/?#@%]|%[0-9A-Fa-f]{{2}})"
    rf"(?:(?:[{_PN_CHARS}.:]|\\[-_~.!$&'()*+,;=/?#@%]|%[0-9A-Fa-f]{{2}})*"
    rf"(?:[{_PN_CHARS}:]|\\[-_~.!$&'()*+,;=/?#@%]|%[0-9A-Fa-f]{{2}}))?)?"
)
_RE_BNODE = re.compile(rf"_:(?P<label>[{_PN_CHARS_U}0-9](?:[{_PN_CHARS}.]*[{_PN_CHARS}])?)")
_RE_LANGTAG = re.compile(r"@(?P<tag>[A-Za-z]+(?:-[A-Za-z0-9]+)*)")
_RE_DOUBLE = re.compile(r"[+-]?(?:\d+\.\d*[eE][+-]?\d+|\.\d+[eE][+-]?\d+|\d+[eE][+-]?\d+)")
_RE_DECIMAL = re.compile(r"[+-]?(?:\d*\.\d+)")
_RE_INTEGER = re.compile(r"[+-]?\d+")

_ESCAPES = {
    "t": "\t", "b": "\b", "n": "\n", "r": "\r", "f": "\f",
    '"': '"', "'": "'", "\\": "\\",
}


@dataclass
class Token:
    kind: str  # IRI PNAME BNODE LITERAL NUMBER DIRECTIVE PUNCT ANON A
    value: object
    line: int
    col: int


class Tokenizer:
    def __init__(self, text: str, source: str = "<string>"):
        self.text = text
        self.pos = 0
        self.line = 1
        self.col = 1
        self.source = source

    def _err(self, msg: str) -> TurtleSyntaxError:
        return TurtleSyntaxError(msg, self.line, self.col, self.source)

    def _advance(self, n: int) -> str:
        chunk = self.text[self.pos : self.pos + n]
        nl = chunk.count("\n")
        if nl:
            self.line += nl
            self.col = len(chunk) - chunk.rfind("\n")
        else:
            self.col += n
        self.pos += n
        return chunk

    def _skip_ws(self) -> None:
        while self.pos < len(self.text):
            ch = self.text[self.pos]
            if ch in " \t\r\n":
                self._advance(1)
            elif ch == "#":
                end = self.text.find("\n", self.pos)
                self._advance((len(self.text) if end == -1 else end) - self.pos)
            else:
                return

    def _read_string_body(self, quote: str, long: bool) -> str:
        out: list[str] = []
        qlen = 3 if long else 1
        while True:
            if self.pos >= len(self.text):
                raise self._err("unterminated string literal")
            if self.text.startswith(quote * qlen, self.pos):
                self._advance(qlen)
                return "".join(out)
            ch = self.text[self.pos]
            if ch == "\\":
                self._advance(1)
                if self.pos >= len(self.text):
                    raise self._err("unterminated escape sequence")
                esc = self.text[self.pos]
                if esc in _ESCAPES:
                    out.append(_ESCAPES[esc])
                    self._advance(1)
                elif esc in "uU":
                    n = 4 if esc == "u" else 8
                    self._advance(1)
                    hexs = self.text[self.pos : self.pos + n]
                    if len(hexs) < n or not all(c in "0123456789abcdefABCDEF" for c in hexs):
                        raise self._err(f"malformed \\{esc} escape")
                    out.append(chr(int(hexs, 16)))
                    self._advance(n)
                else:
                    raise self._err(f"unrecognised escape \\{esc}")
            elif ch == "\n" and not long:
                raise self._err("newline inside single-quoted string; use a triple-quoted literal")
            else:
                out.append(ch)
                self._advance(1)

    def tokens(self) -> Iterator[Token]:
        while True:
            self._skip_ws()
            if self.pos >= len(self.text):
                return
            line, col = self.line, self.col
            ch = self.text[self.pos]

            # IRI reference
            if ch == "<":
                end = self.text.find(">", self.pos)
                if end == -1:
                    raise self._err("unterminated IRI reference")
                raw = self.text[self.pos + 1 : end]
                if any(c in raw for c in " \n\t"):
                    raise self._err(f"whitespace inside IRI reference <{raw[:40]}...>")
                self._advance(end - self.pos + 1)
                # process \u escapes inside IRIs
                raw = re.sub(r"\\u([0-9A-Fa-f]{4})", lambda m: chr(int(m.group(1), 16)), raw)
                yield Token("IRI", raw, line, col)
                continue

            # directives
            if ch == "@":
                m = re.match(r"@(prefix|base)\b", self.text[self.pos :])
                if m:
                    self._advance(m.end())
                    yield Token("DIRECTIVE", m.group(1), line, col)
                    continue
                m = _RE_LANGTAG.match(self.text, self.pos)
                if m:
                    self._advance(m.end() - m.start())
                    yield Token("LANGTAG", m.group("tag"), line, col)
                    continue
                raise self._err("expected @prefix, @base or a language tag after '@'")

            # SPARQL-style directives
            m = re.match(r"(?i:(PREFIX|BASE))\b", self.text[self.pos :])
            if m and ch in "PBpb":
                self._advance(m.end())
                yield Token("DIRECTIVE", m.group(1).lower(), line, col)
                continue

            # strings
            if ch in "\"'":
                if self.text.startswith(ch * 3, self.pos):
                    self._advance(3)
                    yield Token("STRING", self._read_string_body(ch, True), line, col)
                else:
                    self._advance(1)
                    yield Token("STRING", self._read_string_body(ch, False), line, col)
                continue

            # blank node label
            if self.text.startswith("_:", self.pos):
                m = _RE_BNODE.match(self.text, self.pos)
                if not m:
                    raise self._err("malformed blank node label")
                self._advance(m.end() - m.start())
                yield Token("BNODE", m.group("label"), line, col)
                continue

            # punctuation
            if ch in ".;,[]()^":
                if self.text.startswith("^^", self.pos):
                    self._advance(2)
                    yield Token("PUNCT", "^^", line, col)
                    continue
                if ch == ".":
                    # a '.' that begins a decimal is a number, not a terminator
                    m = _RE_DECIMAL.match(self.text, self.pos)
                    if m:
                        self._advance(m.end() - m.start())
                        yield Token("NUMBER", ("decimal", m.group()), line, col)
                        continue
                self._advance(1)
                yield Token("PUNCT", ch, line, col)
                continue

            # numbers  (before PNAME, since '-1' and '1' would otherwise not match anyway)
            for regex, kind in ((_RE_DOUBLE, "double"), (_RE_DECIMAL, "decimal"), (_RE_INTEGER, "integer")):
                m = regex.match(self.text, self.pos)
                if m:
                    # do not swallow the '.' that terminates a statement
                    self._advance(m.end() - m.start())
                    yield Token("NUMBER", (kind, m.group()), line, col)
                    break
            else:
                # keywords and prefixed names
                if re.match(r"a(?![" + _PN_CHARS + r":])", self.text[self.pos :]):
                    self._advance(1)
                    yield Token("A", "a", line, col)
                    continue
                m = re.match(r"(true|false)(?![" + _PN_CHARS + r":])", self.text[self.pos :])
                if m:
                    self._advance(m.end())
                    yield Token("BOOL", m.group(1), line, col)
                    continue
                m = _RE_PNAME.match(self.text, self.pos)
                if m and m.end() > m.start():
                    self._advance(m.end() - m.start())
                    yield Token("PNAME", (m.group("prefix") or "", m.group("local") or ""), line, col)
                    continue
                raise self._err(f"unexpected character {ch!r}")
                continue


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


@dataclass
class Graph:
    triples: list[Triple] = field(default_factory=list)
    prefixes: dict[str, str] = field(default_factory=dict)
    base: str = ""
    source: str = "<string>"

    def add(self, s: Term, p: Term, o: Term) -> None:
        self.triples.append((s, p, o))

    def __len__(self) -> int:
        return len(self.triples)

    def subjects(self, predicate: str | None = None, obj: Term | None = None) -> list[Term]:
        out = []
        for s, p, o in self.triples:
            if predicate is not None and not (isinstance(p, IRI) and p.value == predicate):
                continue
            if obj is not None and o != obj:
                continue
            out.append(s)
        return out

    def objects(self, subject: Term | None = None, predicate: str | None = None) -> list[Term]:
        out = []
        for s, p, o in self.triples:
            if subject is not None and s != subject:
                continue
            if predicate is not None and not (isinstance(p, IRI) and p.value == predicate):
                continue
            out.append(o)
        return out


class TurtleParser:
    def __init__(self, text: str, source: str = "<string>"):
        self.source = source
        self.toks = list(Tokenizer(text, source).tokens())
        self.i = 0
        self.graph = Graph(source=source)
        self._bnode_counter = 0

    # -- token helpers ------------------------------------------------------

    def _peek(self, ahead: int = 0) -> Token | None:
        j = self.i + ahead
        return self.toks[j] if j < len(self.toks) else None

    def _next(self) -> Token:
        if self.i >= len(self.toks):
            last = self.toks[-1] if self.toks else Token("EOF", "", 1, 1)
            raise TurtleSyntaxError("unexpected end of file", last.line, last.col, self.source)
        t = self.toks[self.i]
        self.i += 1
        return t

    def _err(self, tok: Token | None, msg: str) -> TurtleSyntaxError:
        line = tok.line if tok else 0
        col = tok.col if tok else 0
        return TurtleSyntaxError(msg, line, col, self.source)

    def _expect_punct(self, ch: str) -> Token:
        t = self._next()
        if t.kind != "PUNCT" or t.value != ch:
            raise self._err(t, f"expected {ch!r}, found {t.value!r}")
        return t

    def _fresh_bnode(self) -> BNode:
        self._bnode_counter += 1
        return BNode(f"b{self._bnode_counter}")

    # -- IRI resolution -----------------------------------------------------

    def _resolve(self, iri: str) -> str:
        if not self.graph.base or re.match(r"^[A-Za-z][A-Za-z0-9+.\-]*:", iri):
            return iri
        if iri.startswith("#"):
            return self.graph.base + iri
        return self.graph.base.rsplit("/", 1)[0] + "/" + iri if "/" in self.graph.base else self.graph.base + iri

    def _expand_pname(self, tok: Token) -> IRI:
        prefix, local = tok.value
        if prefix not in self.graph.prefixes:
            raise self._err(tok, f"prefix {prefix + ':'!r} used but never declared")
        local = re.sub(r"\\([-_~.!$&'()*+,;=/?#@%])", r"\1", local)
        return IRI(self.graph.prefixes[prefix] + local)

    # -- grammar ------------------------------------------------------------

    def parse(self) -> Graph:
        while self.i < len(self.toks):
            t = self._peek()
            if t.kind == "DIRECTIVE":
                self._directive()
            else:
                self._triples()
        return self.graph

    def _directive(self) -> None:
        d = self._next()
        sparql_style = d.value in ("prefix", "base") and not self._raw_started_with_at(d)
        if d.value == "prefix":
            pt = self._next()
            if pt.kind != "PNAME":
                raise self._err(pt, "expected a prefix name after @prefix")
            prefix = pt.value[0]
            if pt.value[1]:
                raise self._err(pt, "prefix declaration must have an empty local part")
            it = self._next()
            if it.kind != "IRI":
                raise self._err(it, "expected an IRI in prefix declaration")
            self.graph.prefixes[prefix] = self._resolve(it.value)
        else:
            it = self._next()
            if it.kind != "IRI":
                raise self._err(it, "expected an IRI in base declaration")
            self.graph.base = it.value
        if not sparql_style:
            self._expect_punct(".")
        else:
            nxt = self._peek()
            if nxt and nxt.kind == "PUNCT" and nxt.value == ".":
                self._next()

    def _raw_started_with_at(self, tok: Token) -> bool:
        # Turtle-style directives are lowercase in the source; SPARQL-style may be any case.
        # We treat lowercase as @-style, which is how this repository writes them.
        return True

    def _triples(self) -> None:
        t = self._peek()
        if t.kind == "PUNCT" and t.value == "[":
            subject = self._blank_node_property_list()
            nxt = self._peek()
            if nxt and nxt.kind == "PUNCT" and nxt.value == ".":
                self._next()
                return
            self._predicate_object_list(subject)
        else:
            subject = self._term(subject_position=True)
            self._predicate_object_list(subject)
        self._expect_punct(".")

    def _predicate_object_list(self, subject: Term) -> None:
        while True:
            predicate = self._verb()
            self._object_list(subject, predicate)
            nxt = self._peek()
            if nxt and nxt.kind == "PUNCT" and nxt.value == ";":
                self._next()
                nxt2 = self._peek()
                # a trailing ';' before '.' or ']' is legal
                while nxt2 and nxt2.kind == "PUNCT" and nxt2.value == ";":
                    self._next()
                    nxt2 = self._peek()
                if nxt2 and nxt2.kind == "PUNCT" and nxt2.value in ".]":
                    return
                continue
            return

    def _verb(self) -> Term:
        t = self._peek()
        if t.kind == "A":
            self._next()
            return IRI(RDF_TYPE)
        return self._term(subject_position=True)

    def _object_list(self, subject: Term, predicate: Term) -> None:
        while True:
            obj = self._term(subject_position=False)
            self.graph.add(subject, predicate, obj)
            nxt = self._peek()
            if nxt and nxt.kind == "PUNCT" and nxt.value == ",":
                self._next()
                continue
            return

    def _blank_node_property_list(self) -> BNode:
        self._expect_punct("[")
        node = self._fresh_bnode()
        nxt = self._peek()
        if nxt and nxt.kind == "PUNCT" and nxt.value == "]":
            self._next()
            return node
        self._predicate_object_list(node)
        self._expect_punct("]")
        return node

    def _collection(self) -> Term:
        self._expect_punct("(")
        items: list[Term] = []
        while True:
            nxt = self._peek()
            if nxt is None:
                raise self._err(nxt, "unterminated collection")
            if nxt.kind == "PUNCT" and nxt.value == ")":
                self._next()
                break
            items.append(self._term(subject_position=False))
        if not items:
            return IRI(RDF_NIL)
        head = self._fresh_bnode()
        current = head
        for k, item in enumerate(items):
            self.graph.add(current, IRI(RDF_FIRST), item)
            if k == len(items) - 1:
                self.graph.add(current, IRI(RDF_REST), IRI(RDF_NIL))
            else:
                nxt_node = self._fresh_bnode()
                self.graph.add(current, IRI(RDF_REST), nxt_node)
                current = nxt_node
        return head

    def _term(self, subject_position: bool) -> Term:
        t = self._peek()
        if t is None:
            raise self._err(None, "unexpected end of file where a term was expected")

        if t.kind == "IRI":
            self._next()
            return IRI(self._resolve(t.value))
        if t.kind == "PNAME":
            self._next()
            return self._expand_pname(t)
        if t.kind == "BNODE":
            self._next()
            return BNode("label_" + t.value)
        if t.kind == "PUNCT" and t.value == "[":
            return self._blank_node_property_list()
        if t.kind == "PUNCT" and t.value == "(":
            return self._collection()
        if t.kind == "STRING":
            self._next()
            nxt = self._peek()
            if nxt and nxt.kind == "LANGTAG":
                self._next()
                return Literal(t.value, language=nxt.value)
            if nxt and nxt.kind == "PUNCT" and nxt.value == "^^":
                self._next()
                dt = self._next()
                if dt.kind == "IRI":
                    return Literal(t.value, datatype=self._resolve(dt.value))
                if dt.kind == "PNAME":
                    return Literal(t.value, datatype=self._expand_pname(dt).value)
                raise self._err(dt, "expected an IRI after ^^")
            return Literal(t.value)
        if t.kind == "NUMBER":
            self._next()
            kind, raw = t.value
            return Literal(raw, datatype=XSD + kind)
        if t.kind == "BOOL":
            self._next()
            return Literal(t.value, datatype=XSD + "boolean")
        if t.kind == "A" and not subject_position:
            self._next()
            return IRI(RDF_TYPE)
        raise self._err(t, f"unexpected {t.kind} {t.value!r} where a term was expected")


def parse_file(path: str | Path) -> Graph:
    p = Path(path)
    return TurtleParser(p.read_text(encoding="utf-8"), source=str(p)).parse()


def parse_string(text: str, source: str = "<string>") -> Graph:
    return TurtleParser(text, source=source).parse()


def merge(graphs: list[Graph]) -> Graph:
    out = Graph(source="<merged>")
    for k, g in enumerate(graphs):
        for s, p, o in g.triples:
            # keep blank nodes distinct across files
            s2 = BNode(f"g{k}_{s.label}") if isinstance(s, BNode) else s
            o2 = BNode(f"g{k}_{o.label}") if isinstance(o, BNode) else o
            out.add(s2, p, o2)
        out.prefixes.update(g.prefixes)
    return out
