from __future__ import annotations
from collections import defaultdict
import re
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    Iterator,
    Iterable,
    List,
    Optional,
    overload,
    Sequence,
    Type,
    Union,
)
import warnings

from bs4._deprecation import _deprecated
from bs4.element import (
    AttributeDict,
    NavigableString,
    PageElement,
    ResultSet,
    Tag,
)
from bs4._typing import (
    _AtMostOneElement,
    _AttributeValue,
    _OneElement,
    _PageElementMatchFunction,
    _QueryResults,
    _RawAttributeValue,
    _RawAttributeValues,
    _RawOrProcessedAttributeValues,
    _RegularExpressionProtocol,
    _StrainableAttribute,
    _StrainableElement,
    _StrainableString,
    _StringMatchFunction,
    _TagMatchFunction,
)


class ElementFilter(object):
    """`ElementFilter` encapsulates the logic necessary to decide:

    1. whether a `PageElement` (a `Tag` or a `NavigableString`) matches a
    user-specified query.

    2. whether a given sequence of markup found during initial parsing
    should be turned into a `PageElement` at all, or simply discarded.

    The base class is the simplest `ElementFilter`. By default, it
    matches everything and allows all markup to become `PageElement`
    objects. You can make it more selective by passing in a
    user-defined match function, or defining a subclass.

    Most users of Beautiful Soup will never need to use
    `ElementFilter`, or its more capable subclass
    `SoupStrainer`. Instead, they will use methods like
    :py:meth:`Tag.find`, which will convert their arguments into
    `SoupStrainer` objects and run them against the tree.

    However, if you find yourself wanting to treat the arguments to
    Beautiful Soup's find_*() methods as first-class objects, those
    objects will be `SoupStrainer` objects. You can create them
    yourself and then make use of functions like
    `ElementFilter.filter()`.
    """

    match_function: Optional[_PageElementMatchFunction]

    def __init__(self, match_function: Optional[_PageElementMatchFunction] = None):
        """Pass in a match function to easily customize the behavior of
        `ElementFilter.match` without needing to subclass.

        :param match_function: A function that takes a `PageElement`
          and returns `True` if that `PageElement` matches some criteria.
        """
        self.match_function = match_function

    @property
    def excludes_everything(self) -> bool:
        """Does this `ElementFilter` obviously exclude everything? If
        so, Beautiful Soup will issue a warning if you try to use it
        when parsing a document.

        The `ElementFilter` might turn out to exclude everything even
        if this returns `False`, but it won't exclude everything in an
        obvious way.

        The base `ElementFilter` implementation excludes *nothing*, so
        the base implementation of `excludes_everything` always
        returns `False`.
        """
        return False

    def match(self, element: PageElement) -> bool:
        """Does the given PageElement match the rules set down by this
        ElementFilter?

        The base implementation delegates to the function passed in to
        the constructor.
        """
        if not self.match_function:
            return True
        return self.match_function(element)

    def filter(self, generator: Iterator[PageElement]) -> Iterator[_OneElement]:
        """The most generic search method offered by Beautiful Soup.

        Acts like Python's built-in `filter`, using
        `ElementFilter.match` as the filtering function.
        """
        while True:
            try:
                i = next(generator)
            except StopIteration:
                break
            if i:
                if self.match(i):
                    yield cast("_OneElement", i)

    def find(self, generator: Iterator[PageElement]) -> _AtMostOneElement:
        """A lower-level equivalent of :py:meth:`Tag.find`.

        You can pass in your own generator for iterating over
        `PageElement` objects. The first one that matches this
        `ElementFilter` will be returned.

        :param generator: A way of iterating over `PageElement`
            objects.
        """
        for match in self.filter(generator):
            return match
        return None

    def find_all(
        self, generator: Iterator[PageElement], limit: Optional[int] = None
    ) -> _QueryResults:
        """A lower-level equivalent of :py:meth:`Tag.find_all`.

        You can pass in your own generator for iterating over
        `PageElement` objects. Only elements that match this
        `ElementFilter` will be returned in the :py:class:`ResultSet`.

        :param generator: A way of iterating over `PageElement`
            objects.

        :param limit: Stop looking after finding this many results.
        """
        results: _QueryResults = ResultSet(self)
        for match in self.filter(generator):
            results.append(match)
            if limit is not None and len(results) >= limit:
                break
        return results

    def allow_tag_creation(
        self, nsprefix: Optional[str], name: str, attrs: Optional[_RawAttributeValues]
    ) -> bool:
        """Based on the name and attributes of a tag, see whether this
        `ElementFilter` will allow a `Tag` object to even be created.

        By default, all tags are parsed. To change this, subclass
        `ElementFilter`.

        :param name: The name of the prospective tag.
        :param attrs: The attributes of the prospective tag.
        """
        return True

    def allow_string_creation(self, string: str) -> bool:
        """Based on the content of a string, see whether this
        `ElementFilter` will allow a `NavigableString` object based on
        this string to be added to the parse tree.

        By default, all strings are processed into `NavigableString`
        objects. To change this, subclass `ElementFilter`.

        :param str: The string under consideration.
        """
        return True


class MatchRule(object):
    """Each MatchRule encapsulates the logic behind a single argument
    passed in to one of the Beautiful Soup find* methods.
    """

    string: Optional[str]
    pattern: Optional[_RegularExpressionProtocol]
    present: Optional[bool]
    # TODO-TYPING: All MatchRule objects also have an attribute
    # ``function``, but the type of the function depends on the
    # subclass.

    def __init__(
        self,
        string: Optional[Union[str, bytes]] = None,
        pattern: Optional[_RegularExpressionProtocol] = None,
        function: Optional[Callable] = None,
        present: Optional[bool] = None,
    ):
        if isinstance(string, bytes):
            string = string.decode("utf8")
        self.string = string
        if isinstance(pattern, bytes):
            self.pattern = re.compile(pattern.decode("utf8"))
        elif isinstance(pattern, str):
            self.pattern = re.compile(pattern)
        else:
            self.pattern = pattern
        self.function = function
        self.present = present

        values = [
            x
            for x in (self.string, self.pattern, self.function, self.present)
            if x is not None
        ]
        if len(values) == 0:
            raise ValueError(
                "Either string, pattern, function or present must be provided."
            )
        if len(values) > 1:
            raise ValueError(
                "At most one of string, pattern, function and present must be provided."
            )

    def _base_match(self, string: Optional[str]) -> Optional[bool]:
        """Run the 'cheap' portion of a match, trying to get an answer without
        calling a potentially expensive custom function.

        :return: True or False if we have a (positive or negative)
        match; None if we need to keep trying.
        """
        # self.present==True matches everything except None.
        if self.present is True:
            return string is not None

        # self.present==False matches _only_ None.
        if self.present is False:
            return string is None

        # self.string does an exact string match.
        if self.string is not None:
            # print(f"{self.string} ?= {string}")
            return self.string == string

        # self.pattern does a regular expression search.
        if self.pattern is not None:
            # print(f"{self.pattern} ?~ {string}")
            if string is None:
                return False
            return self.pattern.search(string) is not None

        return None

    def matches_string(self, string: Optional[str]) -> bool:
        _base_result = self._base_match(string)
        if _base_result is not None:
            # No need to invoke the test function.
            return _base_result
        if self.function is not None and not self.function(string):
            # print(f"{self.function}({string}) == False")
            return False
        return True

    def __repr__(self) -> str:
        cls = type(self).__name__
        return f"<{cls} string={self.string} pattern={self.pattern} function={self.function} present={self.present}>"

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, MatchRule)
            and self.string == other.string
            and self.pattern == other.pattern
            and self.function == other.function
            and self.present == other.present
        )


class TagNameMatchRule(MatchRule):
    """A MatchRule implementing the rules for matches against tag name."""

    function: Optional[_TagMatchFunction]

    def matches_tag(self, tag: Tag) -> bool:
        base_value = self._base_match(tag.name)
        if base_value is not None:
            return base_value

        # The only remaining possibility is that the match is determined
        # by a function call. Call the function.
        function = cast(_TagMatchFunction, self.function)
        if function(tag):
            return True
        return False


class AttributeValueMatchRule(MatchRule):
    """A MatchRule implementing the rules for matches against attribute value."""

    function: Optional[_StringMatchFunction]


class StringMatchRule(MatchRule):
    """A MatchRule implementing the rules for matches against a NavigableString."""

    function: Optional[_StringMatchFunction]


class SoupStrainer(ElementFilter):
    """The `ElementFilter` subclass used internally by Beautiful Soup.

    A `SoupStrainer` encapsulates the logic necessary to perform the
    kind of matches supported by methods such as
    :py:meth:`Tag.find`. `SoupStrainer` objects are primarily created
    internally, but you can create one yourself and pass it in as
    ``parse_only`` to the `BeautifulSoup` constructor, to parse a
    subset of a large document.

    Internally, `SoupStrainer` objects work by converting the
    constructor arguments into `MatchRule` objects. Incoming
    tags/markup are matched against those rules.

    :param name: One or more restrictions on the tags found in a document.

    :param attrs: A dictionary that maps attribute names to
      restrictions on tags that use those attributes.

    :param string: One or more restrictions on the strings found in a
      document.

    :param kwargs: A dictionary that maps attribute names to restrictions
      on tags that use those attributes. These restrictions are additive to
      any specified in ``attrs``.

    """

    name_rules: List[TagNameMatchRule]
    attribute_rules: Dict[str, List[AttributeValueMatchRule]]
    string_rules: List[StringMatchRule]

    def __init__(
        self,
        name: Optional[_StrainableElement] = None,
        attrs: Dict[str, _StrainableAttribute] = {},
        string: Optional[_StrainableString] = None,
        **kwargs: _StrainableAttribute,
    ):
        if string is None and "text" in kwargs:
            string = cast(Optional[_StrainableString], kwargs.pop("text"))
            warnings.warn(
                "As of version 4.11.0, the 'text' argument to the SoupStrainer constructor is deprecated. Use 'string' instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.name_rules = cast(
            List[TagNameMatchRule], list(self._make_match_rules(name, TagNameMatchRule))
        )
        self.attribute_rules = defaultdict(list)

        if not isinstance(attrs, dict):
            # Passing something other than a dictionary as attrs is
            # sugar for matching that thing against the 'class'
            # attribute.
            attrs = {"class": attrs}

        for attrdict in attrs, kwargs:
            for attr, value in attrdict.items():
                if attr == "class_" and attrdict is kwargs:
                    # If you pass in 'class_' as part of kwargs, it's
                    # because class is a Python reserved word. If you
                    # pass it in as part of the attrs dict, it's
                    # because you really are looking for an attribute
                    # called 'class_'.
                    attr = "class"

                if value is None:
                    value = False
                for rule_obj in self._make_match_rules(value, AttributeValueMatchRule):
                    self.attribute_rules[attr].append(
                        cast(AttributeValueMatchRule, rule_obj)
                    )

        self.string_rules = cast(
            List[StringMatchRule], list(self._make_match_rules(string, StringMatchRule))
        )

        #: DEPRECATED 4.13.0: You shouldn't need to check this under
        #: any name (.string or .text), and if you do, you're probably
        #: not taking into account all of the types of values this
        #: variable might have. Look at the .string_rules list instead.
        self.__string = string

    @property
    def excludes_everything(self) -> bool:
        """Check whether the provided rules will obviously exclude
        everything. (They might exclude everything even if this returns `False`,
        but not in an obvious way.)
        """
        return (
            True
            if (self.string_rules and (self.name_rules or self.attribute_rules))
            else False
        )

    @property
    def string(self) -> Optional[_StrainableString]:
        ":meta private:"
        warnings.warn(
            "Access to deprecated property string. (Look at .string_rules instead) -- Deprecated since version 4.13.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.__string

    @property
    def text(self) -> Optional[_StrainableString]:
        ":meta private:"
        warnings.warn(
            "Access to deprecated property text. (Look at .string_rules instead) -- Deprecated since version 4.13.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.__string

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name_rules} attrs={self.attribute_rules} string={self.string_rules}>"

    @classmethod
    def _make_match_rules(
        cls,
        obj: Optional[Union[_StrainableElement, _StrainableAttribute]],
        rule_class: Type[MatchRule],
    ) -> Iterator[MatchRule]:
        """Convert a vaguely-specific 'object' into one or more well-defined
        `MatchRule` objects.

        :param obj: Some kind of object that corresponds to one or more
           matching rules.
        :param rule_class: Create instances of this `MatchRule` subclass.
        """
        if obj is None:
            return
        if isinstance(obj, (str, bytes)):
            yield rule_class(string=obj)
        elif isinstance(obj, bool):
            yield rule_class(present=obj)
        elif callable(obj):
            yield rule_class(function=obj)
        elif isinstance(obj, _RegularExpressionProtocol):
            yield rule_class(pattern=obj)
        elif hasattr(obj, "__iter__"):
            for o in obj:
                if not isinstance(o, (bytes, str)) and hasattr(o, "__iter__"):
                    # This is almost certainly the user's
                    # mistake. This list contains another list, which
                    # opens up the possibility of infinite
                    # self-reference. In the interests of avoiding
                    # infinite recursion, we'll ignore this item
                    # rather than looking inside.
                    warnings.warn(
                        f"Ignoring nested list {o} to avoid the possibility of infinite recursion.",
                        stacklevel=5,
                    )
                    continue
                for x in cls._make_match_rules(o, rule_class):
                    yield x
        else:
            yield rule_class(string=str(obj))

    def matches_tag(self, tag: Tag) -> bool:
        """Do the rules of this `SoupStrainer` trigger a match against the
        given `Tag`?

        If the `SoupStrainer` has any `TagNameMatchRule`, at least one
        must match the `Tag` or its `Tag.name`.

        If there are any `AttributeValueMatchRule` for a given
        attribute, at least one of them must match the attribute
        value.

        If there are any `StringMatchRule`, at least one must match,
        but a `SoupStrainer` that *only* contains `StringMatchRule`
        cannot match a `Tag`, only a `NavigableString`.
        """
        # String rules cannot not match a Tag on their own.
        if not self.name_rules and not self.attribute_rules:
            return False

        # Optimization for a very common case where the user is
        # searching for a tag with one specific name, and we're
        # looking at a tag with a different name.
        if (
            not tag.prefix
            and len(self.name_rules) == 1
            and self.name_rules[0].string is not None
            and tag.name != self.name_rules[0].string
        ):
            return False

        # If there are name rules, at least one must match. It can
        # match either the Tag object itself or the prefixed name of
        # the tag.
        prefixed_name = None
        if tag.prefix:
            prefixed_name = f"{tag.prefix}:{tag.name}"
        if self.name_rules:
            name_matches = False
            for rule in self.name_rules:
                # attrs = " ".join(
                #     [f"{k}={v}" for k, v in sorted(tag.attrs.items())]
                # )
                # print(f"Testing <{tag.name} {attrs}>{tag.string}</{tag.name}> against {rule}")
                if rule.matches_tag(tag) or (
                    prefixed_name is not None and rule.matches_string(prefixed_name)
                ):
                    name_matches = True
                    break

            if not name_matches:
                return False

        # If there are attribute rules for a given attribute, at least
        # one of them must match. If there are rules for multiple
        # attributes, each attribute must have at least one match.
        for attr, rules in self.attribute_rules.items():
            attr_value = tag.get(attr, None)
            this_attr_match = self._attribute_match(attr_value, rules)
            if not this_attr_match:
                return False

        # If there are string rules, at least one must match.
        if self.string_rules:
            _str = tag.string
            if _str is None:
                return False
            if not self.matches_any_string_rule(_str):
                return False
        return True

    def _attribute_match(
        self,
        attr_value: Optional[_AttributeValue],
        rules: Iterable[AttributeValueMatchRule],
    ) -> bool:
        attr_values: Sequence[Optional[str]]
        if isinstance(attr_value, list):
            attr_values = attr_value
        else:
            attr_values = [cast(str, attr_value)]

        def _match_attribute_value_helper(attr_values: Sequence[Optional[str]]) -> bool:
            for rule in rules:
                for attr_value in attr_values:
                    if rule.matches_string(attr_value):
                        return True
            return False

        this_attr_match = _match_attribute_value_helper(attr_values)
        if not this_attr_match and len(attr_values) > 1:
            # This cast converts Optional[str] to plain str.
            #
            # We know if there's more than one value, there can't be
            # any None in the list, because Beautiful Soup never uses
            # None as a value of a multi-valued attribute, and if None
            # is passed in as attr_value, it's turned into a list with
            # a single element (thus len(attr_values) > 1 fails).
            attr_values = cast(Sequence[str], attr_values)

            # Try again but treat the attribute value
            # as a single string.
            joined_attr_value = " ".join(attr_values)
            this_attr_match = _match_attribute_value_helper([joined_attr_value])
        return this_attr_match

    def allow_tag_creation(
        self, nsprefix: Optional[str], name: str, attrs: Optional[_RawAttributeValues]
    ) -> bool:
        """Based on the name and attributes of a tag, see whether this
        `SoupStrainer` will allow a `Tag` object to even be created.

        :param name: The name of the prospective tag.
        :param attrs: The attributes of the prospective tag.
        """
        if self.string_rules:
            # A SoupStrainer that has string rules can't be used to
            # manage tag creation, because the string rule can't be
            # evaluated until after the tag and all of its contents
            # have been parsed.
            return False
        prefixed_name = None
        if nsprefix:
            prefixed_name = f"{nsprefix}:{name}"
        if self.name_rules:
            # At least one name rule must match.
            name_match = False
            for rule in self.name_rules:
                for x in name, prefixed_name:
                    if x is not None:
                        if rule.matches_string(x):
                            name_match = True
                            break
            if not name_match:
                return False

        # For each attribute that has rules, at least one rule must
        # match.
        if attrs is None:
            attrs = AttributeDict()
        for attr, rules in self.attribute_rules.items():
            attr_value = attrs.get(attr)
            if not self._attribute_match(attr_value, rules):
                return False

        return True

    def allow_string_creation(self, string: str) -> bool:
        """Based on the content of a markup string, see whether this
        `SoupStrainer` will allow it to be instantiated as a
        `NavigableString` object, or whether it should be ignored.
        """
        if self.name_rules or self.attribute_rules:
            # A SoupStrainer that has name or attribute rules won't
            # match any strings; it's designed to match tags with
            # certain properties.
            return False
        if not self.string_rules:
            # A SoupStrainer with no string rules will match
            # all strings.
            return True
        if not self.matches_any_string_rule(string):
            return False
        return True

    def matches_any_string_rule(self, string: str) -> bool:
        """See whether the content of a string matches any of
        this `SoupStrainer`'s string rules.
        """
        if not self.string_rules:
            return True
        for string_rule in self.string_rules:
            if string_rule.matches_string(string):
                return True
        return False

    def match(self, element: PageElement) -> bool:
        """Does the given `PageElement` match the rules set down by this
        `SoupStrainer`?

        The find_* methods rely heavily on this method to find matches.

        :param element: A `PageElement`.
        :return: `True` if the element matches this `SoupStrainer`'s rules; `False` otherwise.
        """
        if isinstance(element, Tag):
            return self.matches_tag(element)
        assert isinstance(element, NavigableString)
        if not (self.name_rules or self.attribute_rules):
            # A NavigableString can only match a SoupStrainer that
            # does not define any name or attribute restrictions.
            for rule in self.string_rules:
                if rule.matches_string(element):
                    return True
        return False

    @_deprecated("allow_tag_creation", "4.13.0")
    def search_tag(self, name: str, attrs: Optional[_RawAttributeValues]) -> bool:
        """A less elegant version of `allow_tag_creation`. Deprecated as of 4.13.0"""
        ":meta private:"
        return self.allow_tag_creation(None, name, attrs)

    @_deprecated("match", "4.13.0")
    def search(self, element: PageElement) -> Optional[PageElement]:
        """A less elegant version of match(). Deprecated as of 4.13.0.

        :meta private:
        """
        return element if self.match(element) else None


def _split_classes(classes: str | Iterable[str] | None) -> list[str]:
    """Converts class specifications into a list of class names.

    Args:
        classes: A single class name, a whitespace-separated string of class
            names, an iterable of class names, or None.

    Returns:
        A list of class name strings. None becomes an empty list.
    """
    if classes is None:
        return []
    if isinstance(classes, str):
        return classes.split()
    return list(classes)


class SoupReplacer(object):
    """Specifies tag name and class attribute changes during document parsing.

    A ``SoupReplacer`` may be constructed in one of three ways:

    1. ``SoupReplacer(og_tag, alt_tag)`` renames every ``og_tag`` to
       ``alt_tag``.

    2. ``SoupReplacer(tag, add_classes, remove_classes)`` adds
       ``add_classes`` to, and removes ``remove_classes`` from, the ``class``
       attribute of every ``tag``.

    3. ``SoupReplacer(name_xformer=..., attrs_xformer=..., xformer=...)``
       applies user-supplied transformer functions to every tag as it is
       parsed. Each transformer takes the :py:class:`bs4.element.Tag` being
       built. ``name_xformer`` returns a replacement tag name and
       ``attrs_xformer`` returns a replacement attribute mapping, while
       ``xformer`` modifies the tag in place and returns nothing.

    Attributes:
        og_tag: The name of the target tag to match during parsing.
        alt_tag: The replacement tag name, or None if the tag name should not
            change.
        add_classes: Classes to add to every matching tag.
        remove_classes: Classes to remove from every matching tag.
        name_xformer: A function that takes a Tag and returns a replacement
            name (or None to keep the name).
        attrs_xformer: A function that takes a Tag and returns a replacement
            attribute mapping.
        xformer: A function that takes a Tag and modifies it in place.
    """

    og_tag: str | None
    alt_tag: str | None
    add_classes: list[str]
    remove_classes: list[str]
    name_xformer: Callable[[Tag], str | None] | None
    attrs_xformer: Callable[[Tag], _RawOrProcessedAttributeValues] | None
    xformer: Callable[[Tag], None] | None
    _name_mapping: Dict[str, str]

    @overload
    def __init__(self, og_tag: str, alt_tag: str) -> None:
        ...

    @overload
    def __init__(
        self,
        tag: str,
        add_classes: str | Iterable[str],
        remove_classes: str | Iterable[str],
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        *,
        name_xformer: Callable[[Tag], str | None] | None = None,
        attrs_xformer: Callable[[Tag], _RawOrProcessedAttributeValues] | None = None,
        xformer: Callable[[Tag], None] | None = None,
    ) -> None:
        ...

    def __init__(
        self,
        og_tag: str | None = None,
        *args: Any,
        alt_tag: str | None = None,
        add_classes: str | Iterable[str] | None = None,
        remove_classes: str | Iterable[str] | None = None,
        name_xformer: Callable[[Tag], str | None] | None = None,
        attrs_xformer: Callable[[Tag], _RawOrProcessedAttributeValues] | None = None,
        xformer: Callable[[Tag], None] | None = None,
    ):
        """Initializes a SoupReplacer for tag renaming and/or class modification.

        The constructor may be called in one of three ways:

        1. ``SoupReplacer(og_tag, alt_tag)`` renames every ``og_tag`` to
           ``alt_tag``.

        2. ``SoupReplacer(tag, add_classes, remove_classes)`` adds
           ``add_classes`` to, and removes ``remove_classes`` from, the
           ``class`` attribute of every ``tag``.

        3. ``SoupReplacer(name_xformer, attrs_xformer, xformer)`` applies
           transformer functions to every parsed tag. The transformers take
           the tag being built as their argument. ``name_xformer`` returns a
           replacement for the tag's name, ``attrs_xformer`` returns a
           replacement for the tag's attributes, and ``xformer`` performs
           arbitrary in-place changes to the tag and returns nothing.

        Args:
            og_tag: The name of the tag to match.
            alt_tag: The new name for matching tags. Defaults to None (keep
                the original tag name).
            add_classes: Classes to add to matching tags as a whitespace
                string or iterable of strings. Defaults to None.
            remove_classes: Classes to remove from matching tags as a
                whitespace string or iterable of strings, or "*" to remove all
                classes. Defaults to None.
            name_xformer: A function that takes a Tag and returns the name it
                should have (or None to keep its current name).
            attrs_xformer: A function that takes a Tag and returns the
                attribute mapping it should have.
            xformer: A function that takes a Tag and modifies it in place.
                Its return value is ignored.
        """
        if len(args) == 1:
            alt_tag = args[0]
        elif len(args) == 2:
            add_classes, remove_classes = args
        elif len(args) > 2:
            raise TypeError(
                "SoupReplacer takes at most 3 positional arguments "
                "(og_tag, alt_tag, add_classes, remove_classes)."
            )
        if (
            og_tag is None
            and name_xformer is None
            and attrs_xformer is None
            and xformer is None
        ):
            raise TypeError(
                "SoupReplacer requires either a tag name (og_tag) or at "
                "least one of name_xformer, attrs_xformer, or xformer."
            )
        self.og_tag = og_tag
        self.alt_tag = alt_tag
        self.add_classes = _split_classes(add_classes)
        self.remove_classes = _split_classes(remove_classes)
        self.name_xformer = name_xformer
        self.attrs_xformer = attrs_xformer
        self.xformer = xformer
        self._name_mapping = {}

    @classmethod
    def for_tag_replacement(cls, og_tag: str, alt_tag: str) -> SoupReplacer:
        """Creates a SoupReplacer that renames matching tags.

        Args:
            og_tag: The name of the tag to be replaced.
            alt_tag: The name the tag will be replaced with.

        Returns:
            A SoupReplacer configured to rename og_tag to alt_tag.
        """
        return cls(og_tag=og_tag, alt_tag=alt_tag)

    @classmethod
    def for_class_replacement(
        cls,
        tag: str,
        add_classes: str | Iterable[str] | None = None,
        remove_classes: str | Iterable[str] | None = None,
    ) -> SoupReplacer:
        """Creates a SoupReplacer that modifies the class attribute of tags.

        Args:
            tag: The name of the tag whose class attribute will be modified.
            add_classes: Classes to add to every matching tag.
            remove_classes: Classes to remove from every matching tag, or "*"
                to remove all classes.

        Returns:
            A SoupReplacer configured to modify classes on matching tags.
        """
        return cls(
            og_tag=tag,
            alt_tag=None,
            add_classes=add_classes,
            remove_classes=remove_classes,
        )

    def replace_tag_name(self, name: str) -> str:
        """Returns the tag name to use for a parsed tag.

        This handles the string-based ``og_tag``/``alt_tag`` rename and is
        called when a start tag is encountered. Tag-based ``name_xformer``
        renames happen later, in :meth:`replace_tag`.

        Args:
            name: The parsed name of the tag.

        Returns:
            alt_tag if name matches og_tag and alt_tag is set; otherwise name.
        """
        if name == self.og_tag and self.alt_tag is not None:
            return self.alt_tag
        return name

    def replace_end_tag_name(self, name: str) -> str:
        """Returns the tag name to match against for a parsed end tag.

        A start tag that was renamed by ``name_xformer`` must have its end
        tag renamed the same way, so the tree builder can find the matching
        open tag on the stack.

        Args:
            name: The parsed name of the end tag.

        Returns:
            The name to look for when closing this end tag.
        """
        name = self.replace_tag_name(name)
        if self.name_xformer is not None:
            name = self._name_mapping.get(name, name)
        return name

    def replace_tag(self, tag: Tag) -> None:
        """Applies any transformer functions to an already-created Tag.

        This is called by ``BeautifulSoup`` after a ``Tag`` has been created
        during parsing, but before it is added to the tree, so the
        transformer functions can observe and modify the tag in place.

        Args:
            tag: The Tag that was just created for a parsed start tag.
        """
        if self.xformer is not None:
            self.xformer(tag)

        if self.name_xformer is not None:
            original_name = tag.name
            new_name = self.name_xformer(tag)
            if new_name is not None and new_name != original_name:
                self._name_mapping[original_name] = new_name
                tag.name = new_name

        if self.attrs_xformer is not None:
            new_attrs = self.attrs_xformer(tag)
            tag.attrs = type(tag.attrs)(new_attrs)

    def replace_class(self, name: str, attrs: _RawAttributeValues) -> None:
        """Modifies the class attribute of attrs in place if name matches.

        Args:
            name: The original name of the tag being processed.
            attrs: Mutable mapping of the tag's raw attribute values.
        """
        if name != self.og_tag:
            return
        if not self.add_classes and not self.remove_classes:
            return

        raw_value: _RawAttributeValue | None = attrs.get("class")
        classes: list[str]
        if isinstance(raw_value, str):
            classes = raw_value.split()
        elif raw_value is None:
            classes = []
        else:
            # The class attribute might already have been converted into a
            # list before reaching this method.
            classes = list(raw_value)

        if self.remove_classes:
            if self.remove_classes == ["*"]:
                classes = []
            else:
                classes = [
                    cls for cls in classes if cls not in self.remove_classes
                ]
        for cls in self.add_classes:
            if cls not in classes:
                classes.append(cls)

        if raw_value is not None or classes:
            attrs["class"] = " ".join(classes)

    def __repr__(self) -> str:
        if self.og_tag is None:
            tag_part = (
                f"name_xformer={self.name_xformer} "
                f"attrs_xformer={self.attrs_xformer} "
                f"xformer={self.xformer}"
            )
        else:
            tag_part = (
                f"{self.og_tag}->{self.alt_tag}"
                if self.alt_tag is not None
                else f"tag={self.og_tag}"
            )
        return (
            f"<{self.__class__.__name__} {tag_part} "
            f"add_classes={self.add_classes} "
            f"remove_classes={self.remove_classes}>"
        )
