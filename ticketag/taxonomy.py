"""The closed set of labels a ticket can be assigned to."""

from __future__ import annotations

from dataclasses import dataclass

from .exceptions import ConfigurationError


@dataclass(frozen=True, slots=True)
class Category:
    """A label the classifier is allowed to return."""

    name: str
    description: str = ""

    def as_prompt_line(self) -> str:
        """Render the category as a single bullet for the prompt."""
        return f"- {self.name}: {self.description}" if self.description else f"- {self.name}"


DEFAULT_CATEGORIES: tuple[Category, ...] = (
    Category("Hardware", "Physical equipment: laptops, monitors, printers, cables, peripherals."),
    Category("HR Support", "People topics: onboarding, payroll, leave, benefits, contracts."),
    Category("Access", "Accounts, passwords, VPN, licenses and permissions to use a system."),
    Category("Miscellaneous", "Generic questions or requests that fit no other category."),
    Category("Storage", "Disk space, shared folders, backups, file servers and quotas."),
    Category("Purchase", "Buying goods or services: quotes, orders, invoices, vendors."),
    Category("Internal Project", "Work tracked as a project: rollouts, migrations, deployments."),
    Category("Administrative rights", "Elevated or admin privileges on a machine or application."),
)


def parse_categories(values: list[str] | None) -> tuple[Category, ...]:
    """Turn CLI 'Name: description' strings into categories, defaulting to the taxonomy."""
    if not values:
        return DEFAULT_CATEGORIES
    categories = []
    for value in values:
        name, _, description = value.partition(":")
        if not name.strip():
            raise ConfigurationError(f"Invalid category definition: {value!r}")
        categories.append(Category(name.strip(), description.strip()))
    return tuple(categories)
