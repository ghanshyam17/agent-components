"""Ready-made schemas for the document types people actually process.

Each is a plain :class:`DocumentSchema`, so it can be copied, extended or
diffed like any other data. They exist because "what does a valid invoice look
like?" is domain knowledge worth writing down once — and because the cross-field
rules here (subtotal + tax ≈ total, invoice date ≤ due date) are the checks that
catch real errors, which are easy to omit when building a schema by hand.
"""
from __future__ import annotations

from doc_processing.models import (
    CrossFieldRule,
    DocumentSchema,
    FieldRule,
    Severity,
)

__all__ = ["INVOICE_SCHEMA", "RECEIPT_SCHEMA", "CONTRACT_SCHEMA", "ID_DOCUMENT_SCHEMA", "PRESETS"]

#: Monetary tolerance, in major units. One cent: invoices are usually exact,
#: but a rounding line means a strict equality check produces false failures.
_MONEY_TOL = 0.02

INVOICE_SCHEMA = DocumentSchema(
    name="invoice",
    description="Supplier invoice: header fields, amounts, and reconciliation rules.",
    prebuilt_model="prebuilt-invoice",
    fields=[
        FieldRule(name="invoice_number", required=True, type="string", min_length=2, min_confidence=0.5),
        FieldRule(name="invoice_date", required=True, type="date", min_confidence=0.5),
        FieldRule(name="due_date", type="date", min_confidence=0.5),
        FieldRule(name="vendor_name", required=True, type="string", min_length=2, min_confidence=0.5),
        FieldRule(name="vendor_tax_id", type="string", min_confidence=0.4),
        FieldRule(name="customer_name", type="string", min_confidence=0.4),
        FieldRule(name="po_number", type="string", min_confidence=0.4),
        FieldRule(name="currency", type="string", allowed_values=["EUR", "USD", "GBP", "CAD", "CHF"], min_confidence=0.3),
        FieldRule(name="subtotal", type="number", min_value=0.0, min_confidence=0.5),
        FieldRule(name="tax", type="number", min_value=0.0, min_confidence=0.4),
        FieldRule(name="total", required=True, type="number", min_value=0.0, min_confidence=0.5),
    ],
    cross_field=[
        CrossFieldRule(
            name="amounts_reconcile",
            description="subtotal + tax must equal total.",
            kind="sum_equals",
            fields=["subtotal", "tax"],
            target="total",
            tolerance=_MONEY_TOL,
            severity=Severity.ERROR,
        ),
        CrossFieldRule(
            name="dates_ordered",
            description="An invoice cannot be due before it was issued.",
            kind="date_ordering",
            fields=["invoice_date", "due_date"],
            severity=Severity.ERROR,
        ),
        CrossFieldRule(
            name="po_requires_customer",
            description="A PO number implies a named customer.",
            kind="implies",
            fields=["po_number", "customer_name"],
            severity=Severity.WARNING,
        ),
        CrossFieldRule(
            name="has_an_amount",
            description="At least one of subtotal/total must be present.",
            kind="one_of",
            fields=["subtotal", "total"],
            severity=Severity.ERROR,
        ),
    ],
    hints={
        "invoice_number": ["invoice no", "invoice #", "facture", "rechnungsnummer", "inv no"],
        "invoice_date": ["invoice date", "date of issue", "issued", "date"],
        "due_date": ["due date", "payment due", "échéance", "due"],
        "vendor_name": ["vendor", "supplier", "from", "fournisseur", "seller"],
        "customer_name": ["customer", "bill to", "client", "buyer"],
        "po_number": ["po", "purchase order", "order no", "po number"],
        "subtotal": ["subtotal", "net amount", "sous-total"],
        "tax": ["vat", "tva", "tax", "gst"],
        "total": ["total", "amount due", "total ttc", "grand total"],
        "currency": ["currency", "devise"],
    },
)

RECEIPT_SCHEMA = DocumentSchema(
    name="receipt",
    description="Point-of-sale receipt: merchant, date, total, payment method.",
    fields=[
        FieldRule(name="merchant_name", required=True, type="string", min_length=2, min_confidence=0.4),
        FieldRule(name="transaction_date", required=True, type="date", min_confidence=0.4),
        FieldRule(name="total", required=True, type="number", min_value=0.0, min_confidence=0.4),
        FieldRule(name="tax", type="number", min_value=0.0, min_confidence=0.3),
        FieldRule(name="payment_method", type="string", allowed_values=["cash", "card", "credit", "debit", "mobile"], min_confidence=0.3),
    ],
    cross_field=[
        CrossFieldRule(
            name="total_at_least_tax",
            description="The total must be at least the tax component.",
            kind="gte",
            fields=["total", "tax"],
            severity=Severity.WARNING,
        ),
    ],
    hints={
        "merchant_name": ["merchant", "store", "shop", "magasin"],
        "transaction_date": ["date", "time"],
        "total": ["total", "amount", "montant"],
        "tax": ["tax", "vat", "tva"],
        "payment_method": ["payment", "paid by", "card", "cash", "paiement"],
    },
)

CONTRACT_SCHEMA = DocumentSchema(
    name="contract",
    description="Contract header: parties, term dates, value and governing law.",
    prebuilt_model="prebuilt-contract",
    fields=[
        FieldRule(name="party_a", required=True, type="string", min_length=2, min_confidence=0.4),
        FieldRule(name="party_b", required=True, type="string", min_length=2, min_confidence=0.4),
        FieldRule(name="effective_date", required=True, type="date", min_confidence=0.4),
        FieldRule(name="expiry_date", type="date", min_confidence=0.4),
        FieldRule(name="contract_value", type="number", min_value=0.0, min_confidence=0.3),
        FieldRule(name="governing_law", type="string", min_confidence=0.3),
    ],
    cross_field=[
        CrossFieldRule(
            name="term_ordered",
            description="A contract cannot expire before it takes effect.",
            kind="date_ordering",
            fields=["effective_date", "expiry_date"],
            severity=Severity.ERROR,
        ),
        CrossFieldRule(
            name="value_implies_expiry",
            description="A stated value implies a defined term.",
            kind="implies",
            fields=["contract_value", "expiry_date"],
            severity=Severity.WARNING,
        ),
    ],
    hints={
        "party_a": ["between", "party a", "contractor", "provider"],
        "party_b": ["and", "party b", "client", "customer"],
        "effective_date": ["effective date", "commencement", "start date", "entered into"],
        "expiry_date": ["expiry", "expiration", "end date", "term"],
        "contract_value": ["value", "amount", "fee", "consideration"],
        "governing_law": ["governing law", "jurisdiction", "governed by"],
    },
)

ID_DOCUMENT_SCHEMA = DocumentSchema(
    name="id_document",
    description="Identity document: name, number, dates. Sensitive — expect PII redaction upstream.",
    prebuilt_model="prebuilt-idDocument",
    fields=[
        FieldRule(name="document_number", required=True, type="string", min_length=3, min_confidence=0.5),
        FieldRule(name="first_name", type="string", min_confidence=0.4),
        FieldRule(name="last_name", required=True, type="string", min_confidence=0.4),
        FieldRule(name="date_of_birth", type="date", min_confidence=0.4),
        FieldRule(name="expiry_date", type="date", min_confidence=0.4),
        FieldRule(name="nationality", type="string", min_confidence=0.3),
    ],
    cross_field=[
        CrossFieldRule(
            name="not_expired_ordering",
            description="Date of birth must precede expiry.",
            kind="date_ordering",
            fields=["date_of_birth", "expiry_date"],
            severity=Severity.ERROR,
        ),
        CrossFieldRule(
            name="has_a_name",
            description="At least one name field must be present.",
            kind="one_of",
            fields=["first_name", "last_name"],
            severity=Severity.ERROR,
        ),
    ],
    hints={
        "document_number": ["document no", "passport no", "id number", "licence no"],
        "first_name": ["given name", "first name", "prénom"],
        "last_name": ["surname", "last name", "nom"],
        "date_of_birth": ["date of birth", "dob", "naissance", "born"],
        "expiry_date": ["expiry", "expires", "valid until", "date of expiry"],
        "nationality": ["nationality", "citizen"],
    },
)

#: Schema name → schema, for CLI/API lookup.
PRESETS: dict[str, DocumentSchema] = {
    s.name: s for s in (INVOICE_SCHEMA, RECEIPT_SCHEMA, CONTRACT_SCHEMA, ID_DOCUMENT_SCHEMA)
}
