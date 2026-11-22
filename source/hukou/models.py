"""Serializable household evidence and review records."""
from dataclasses import dataclass,field,asdict

@dataclass
class Issue:
    field: str
    reason: str
    value: str=''
    page: int=0
    member: str=''

@dataclass
class Member:
    name: str=''
    relation: str=''
    id_number: str=''
    registration_date: str=''
    source_page: int=0
    region_index: int=0
    confidence: dict=field(default_factory=dict)
    raw: dict=field(default_factory=dict)
    evidence: dict=field(default_factory=dict)
    fingerprint: str=''
    # VALID | UNCERTAIN | CONFLICT | MISSING per field key.
    field_status: dict=field(default_factory=dict)

@dataclass
class Household:
    source_pdf: str=''
    source_hash: str=''
    household_no: str=''
    householder: str=''
    address: str=''
    members: list[Member]=field(default_factory=list)
    family_size: int|None=None
    valid_member_count: int=0
    page_count: int=0
    status: str='review'
    issues: list[Issue]=field(default_factory=list)
    pages: list=field(default_factory=list)
    covers: list=field(default_factory=list)
    rename_reliable: bool=False
    new_pdf: str=''
    elapsed_seconds: float=0.
    field_evidence: dict=field(default_factory=dict)
    identity_cards: list=field(default_factory=list)
    # Internal grouping only; never exported to customer Excel as an ID.
    household_groups: list=field(default_factory=list)
    primary_group_index: int=0
    field_status: dict=field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls,value: dict) -> 'Household':
        value=dict(value)
        value['members']=[Member(**m) for m in value.get('members',[])]
        value['issues']=[Issue(**i) for i in value.get('issues',[])]
        return cls(**value)
