"""Graph tools for Sangue e Grafi — RLM-on-KG action space.

Provides four core graph operations via two backends:
  - WoGraphTools: calls WoGraph REST / GraphQL API (production)
  - RdflibTools:  operates on a local rdflib.Graph (HF Space demo / testing)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import httpx
from rdflib import Graph, Literal, Namespace, RDF, URIRef
from rdflib.term import Node

# Canonical namespace for the kinship ontology
EX = Namespace("http://example.org/kinship#")


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class GraphToolsBase(ABC):
    """Four-tool action space consumed by the reasoning agent."""

    @abstractmethod
    def search_entities(self, query: str) -> list[dict]:
        """Fuzzy-search entities by name.

        Args:
            query: Free-text search string (matched against fullName).

        Returns:
            List of dicts, each containing at least ``iri`` and ``label``.
        """
        ...

    @abstractmethod
    def get_entity(self, entity_iri: str) -> dict:
        """Retrieve full details for a single entity.

        Args:
            entity_iri: The IRI (URI) of the entity.

        Returns:
            Dict with all known properties for the entity.
        """
        ...

    @abstractmethod
    def follow_entity_link(
        self,
        entity_iri: str,
        property_uri: str | None = None,
    ) -> list[dict]:
        """Follow outbound object-property links from an entity.

        Args:
            entity_iri: Source entity IRI.
            property_uri: Optional property IRI to filter by (e.g.
                ``http://example.org/kinship#isBiologicalParentOf``).
                When *None*, all outbound object-property links are returned.

        Returns:
            List of dicts with ``property``, ``target_iri``, ``target_label``.
        """
        ...

    @abstractmethod
    def expand_neighbors(self, entity_iri: str) -> dict:
        """Return both inbound and outbound links for an entity.

        Args:
            entity_iri: The entity to expand.

        Returns:
            Dict with keys ``outbound`` and ``inbound``, each a list of dicts
            containing ``property``, ``target_iri``, ``target_label``.
        """
        ...


# ---------------------------------------------------------------------------
# WoGraph backend (production)
# ---------------------------------------------------------------------------

class WoGraphTools(GraphToolsBase):
    """Calls WoGraph's REST / GraphQL API via httpx.

    Args:
        endpoint: Base URL of the WoGraph instance (e.g.
            ``https://api.wograph.io``).
        api_key: Bearer token or API key for authentication.
    """

    def __init__(self, endpoint: str, api_key: str) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self._client = httpx.Client(
            base_url=self.endpoint,
            headers={
                "Authorization": f"Key {api_key}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )

    # -- helpers -------------------------------------------------------------

    def _graphql(self, query: str, variables: dict[str, Any] | None = None) -> dict:
        """Execute a GraphQL query against the WoGraph endpoint."""
        payload: dict[str, Any] = {"query": query}
        if variables:
            payload["variables"] = variables
        resp = self._client.post("/graphql", json=payload)
        resp.raise_for_status()
        body = resp.json()
        if "errors" in body:
            raise RuntimeError(f"WoGraph GraphQL errors: {body['errors']}")
        return body.get("data", {})

    # -- tools ---------------------------------------------------------------

    def search_entities(self, query: str) -> list[dict]:
        """Search entities via WoGraph's ``entitySearch`` GraphQL resolver."""
        gql = """
        query SearchEntities($query: String!) {
            entitySearch(query: $query) {
                iri
                label
                types
            }
        }
        """
        data = self._graphql(gql, {"query": query})
        results = data.get("entitySearch", [])
        return [
            {
                "iri": r["iri"],
                "label": r.get("label", ""),
                "types": r.get("types", []),
            }
            for r in results
        ]

    def get_entity(self, entity_iri: str) -> dict:
        """Fetch a single entity as JSON-LD via ``GET /entities``."""
        resp = self._client.get("/entities", params={"id": entity_iri})
        resp.raise_for_status()
        return resp.json()

    def follow_entity_link(
        self,
        entity_iri: str,
        property_uri: str | None = None,
    ) -> list[dict]:
        """Resolve outbound references via WoGraph GraphQL ``refs`` resolver."""
        if property_uri:
            gql = """
            query FollowLink($iri: String!, $property: String!) {
                entity(iri: $iri) {
                    ref(property: $property) {
                        iri
                        label
                    }
                }
            }
            """
            data = self._graphql(gql, {"iri": entity_iri, "property": property_uri})
            entity_data = data.get("entity", {})
            refs = entity_data.get("ref", [])
            if isinstance(refs, dict):
                refs = [refs]
            return [
                {
                    "property": property_uri,
                    "target_iri": r["iri"],
                    "target_label": r.get("label", ""),
                }
                for r in refs
            ]
        else:
            gql = """
            query FollowAllLinks($iri: String!) {
                entity(iri: $iri) {
                    refs {
                        property
                        target {
                            iri
                            label
                        }
                    }
                }
            }
            """
            data = self._graphql(gql, {"iri": entity_iri})
            entity_data = data.get("entity", {})
            refs = entity_data.get("refs", [])
            return [
                {
                    "property": r["property"],
                    "target_iri": r["target"]["iri"],
                    "target_label": r["target"].get("label", ""),
                }
                for r in refs
            ]

    def expand_neighbors(self, entity_iri: str) -> dict:
        """Combine outbound refs + SPARQL query for inbound links."""
        # Outbound via GraphQL
        outbound = self.follow_entity_link(entity_iri)

        # Inbound via SPARQL endpoint
        sparql_query = f"""
        SELECT ?s ?p WHERE {{
            ?s ?p <{entity_iri}> .
            FILTER(?p != <{RDF.type}>)
        }}
        """
        try:
            resp = self._client.post(
                "/sparql",
                json={"query": sparql_query},
            )
            resp.raise_for_status()
            sparql_results = resp.json().get("results", {}).get("bindings", [])
        except (httpx.HTTPError, KeyError):
            sparql_results = []

        inbound = [
            {
                "property": b["p"]["value"],
                "target_iri": b["s"]["value"],
                "target_label": "",
            }
            for b in sparql_results
        ]
        return {"outbound": outbound, "inbound": inbound}

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> "WoGraphTools":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# rdflib backend (local / demo)
# ---------------------------------------------------------------------------

class RdflibTools(GraphToolsBase):
    """Operates on a local ``rdflib.Graph`` instance.

    Args:
        graph: An ``rdflib.Graph`` already populated with kinship data.
    """

    def __init__(self, graph: Graph) -> None:
        self.graph = graph

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _label_for(node: Node, graph: Graph) -> str:
        """Best-effort human label for *node*."""
        for label in graph.objects(node, EX.fullName):
            return str(label)
        for label in graph.objects(node, EX.name):
            return str(label)
        # Fallback: local part of IRI
        local = str(node).rsplit("#", 1)[-1].rsplit("/", 1)[-1]
        return local

    # -- tools ---------------------------------------------------------------

    def search_entities(self, query: str) -> list[dict]:
        """SPARQL text match on ``ex:fullName``.

        Uses a case-insensitive ``CONTAINS`` filter.
        """
        sparql = """
        SELECT DISTINCT ?entity ?name WHERE {
            ?entity ex:fullName ?name .
            FILTER(CONTAINS(LCASE(?name), LCASE(?query)))
        }
        """
        # rdflib's preparedQuery doesn't support parameter injection in
        # FILTER strings easily, so we do manual safe substitution.
        safe_query = query.replace("\\", "\\\\").replace('"', '\\"')
        sparql = sparql.replace("?query", f'"{safe_query}"')
        results: list[dict] = []
        for row in self.graph.query(sparql, initNs={"ex": EX}):
            results.append(
                {
                    "iri": str(row.entity),
                    "label": str(row.name),
                }
            )
        return results

    def get_entity(self, entity_iri: str) -> dict:
        """Collect every triple where *entity_iri* is the subject."""
        entity = URIRef(entity_iri)
        props: dict[str, Any] = {"iri": entity_iri}
        for pred, obj in self.graph.predicate_objects(entity):
            key = str(pred).rsplit("#", 1)[-1].rsplit("/", 1)[-1]
            value: str | int | float | bool
            if isinstance(obj, Literal):
                value = obj.toPython()
            else:
                value = str(obj)
            # Accumulate multi-valued properties as lists
            if key in props:
                existing = props[key]
                if isinstance(existing, list):
                    existing.append(value)
                else:
                    props[key] = [existing, value]
            else:
                props[key] = value
        return props

    def follow_entity_link(
        self,
        entity_iri: str,
        property_uri: str | None = None,
    ) -> list[dict]:
        """SPARQL SELECT for outbound object properties, optionally filtered."""
        entity = URIRef(entity_iri)
        if property_uri:
            prop = URIRef(property_uri)
            sparql = """
            SELECT ?target WHERE {
                ?entity ?prop ?target .
                FILTER(isIRI(?target))
            }
            """
            results: list[dict] = []
            for row in self.graph.query(
                sparql,
                initBindings={"entity": entity, "prop": prop},
            ):
                results.append(
                    {
                        "property": property_uri,
                        "target_iri": str(row.target),
                        "target_label": self._label_for(row.target, self.graph),
                    }
                )
            return results
        else:
            sparql = """
            SELECT ?prop ?target WHERE {
                ?entity ?prop ?target .
                FILTER(isIRI(?target))
            }
            """
            results = []
            for row in self.graph.query(
                sparql,
                initBindings={"entity": entity},
            ):
                results.append(
                    {
                        "property": str(row.prop),
                        "target_iri": str(row.target),
                        "target_label": self._label_for(row.target, self.graph),
                    }
                )
            return results

    def expand_neighbors(self, entity_iri: str) -> dict:
        """SPARQL for both inbound and outbound, excluding ``rdf:type``."""
        entity = URIRef(entity_iri)

        # Outbound
        out_sparql = """
        SELECT ?prop ?target WHERE {
            ?entity ?prop ?target .
            FILTER(isIRI(?target))
            FILTER(?prop != rdf:type)
        }
        """
        outbound: list[dict] = []
        for row in self.graph.query(
            out_sparql,
            initBindings={"entity": entity},
            initNs={"rdf": RDF},
        ):
            outbound.append(
                {
                    "property": str(row.prop),
                    "target_iri": str(row.target),
                    "target_label": self._label_for(row.target, self.graph),
                }
            )

        # Inbound
        in_sparql = """
        SELECT ?source ?prop WHERE {
            ?source ?prop ?entity .
            FILTER(?prop != rdf:type)
        }
        """
        inbound: list[dict] = []
        for row in self.graph.query(
            in_sparql,
            initBindings={"entity": entity},
            initNs={"rdf": RDF},
        ):
            inbound.append(
                {
                    "property": str(row.prop),
                    "target_iri": str(row.source),
                    "target_label": self._label_for(row.source, self.graph),
                }
            )

        return {"outbound": outbound, "inbound": inbound}
