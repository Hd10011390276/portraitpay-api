"""
PortraitPay AI - Admin Automation Module
Single-founder operation automation: auto-triage, reports, health checks
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)

# Import database helpers
try:
    from portrait_db import get_db_conn, last_insert_id
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    logger.warning("portrait_db not available, admin automation will use mock data")


class TicketPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class TicketCategory(Enum):
    AUTH_ISSUE = "auth_issue"           # Login/password problems
    AUTHORIZATION = "authorization"        # Licensing questions
    DISPUTE = "dispute"                  # Rights disputes
    ENTERPRISE = "enterprise"            # Business inquiries
    BUG_REPORT = "bug_report"            # Technical issues
    DATA_REQUEST = "data_request"        # GDPR/data export
    PAYMENT = "payment"                  # Payment issues
    OTHER = "other"


class AdminAutomation:
    """
    Automated admin operations for single-founder operation.
    Handles auto-triage, report generation, and system health.
    """

    def __init__(self):
        self.ticket_keywords = {
            TicketCategory.AUTH_ISSUE: ['密码', '登录', '账户', '找回', '忘记', '登录不上', 'password', 'login', 'account'],
            TicketCategory.AUTHORIZATION: ['授权', '许可', '版权', '使用', '授权范围', 'license', 'permission', 'copyright', 'authorize'],
            TicketCategory.DISPUTE: ['侵权', '投诉', '争议', '盗用', '未经授权', 'dispute', 'infringement', 'complaint', 'unauthorized'],
            TicketCategory.ENTERPRISE: ['企业', '合作', '批量', 'API', 'enterprise', 'business', 'partnership', 'bulk'],
            TicketCategory.BUG_REPORT: ['bug', '错误', '崩溃', '不能', '无法', '问题', 'error', 'crash', 'issue'],
            TicketCategory.DATA_REQUEST: ['导出', '删除', '数据', '隐私', 'export', 'delete', 'data', 'privacy', 'gdpr'],
            TicketCategory.PAYMENT: ['支付', '收益', '提现', '到账', 'payment', 'withdraw', 'earning', 'revenue'],
        }

    def triage_message(self, subject: str, body: str, email: str = "") -> Dict[str, Any]:
        """
        Auto-triage an incoming message into categories and priority.
        Returns category, priority, and recommended action.
        """
        text = f"{subject} {body}".lower()

        # Determine category
        category_scores = {}
        for category, keywords in self.ticket_keywords.items():
            score = sum(1 for kw in keywords if kw.lower() in text)
            if score > 0:
                category_scores[category] = score

        if category_scores:
            category = max(category_scores, key=category_scores.get)
        else:
            category = TicketCategory.OTHER

        # Determine priority
        priority = self._calculate_priority(category, text, email)

        # Determine recommended action
        action = self._get_recommended_action(category, priority)

        return {
            "category": category.value,
            "priority": priority.value,
            "action": action,
            "confidence": category_scores.get(category, 0) / max(sum(category_scores.values()), 1) if category_scores else 0.5,
            "auto_assigned": True
        }

    def _calculate_priority(self, category: TicketCategory, text: str, email: str) -> TicketPriority:
        """Calculate ticket priority based on category and content."""
        # High priority categories
        if category in [TicketCategory.DISPUTE, TicketCategory.PAYMENT]:
            base_priority = TicketPriority.HIGH
        elif category in [TicketCategory.AUTH_ISSUE, TicketCategory.DATA_REQUEST]:
            base_priority = TicketPriority.MEDIUM
        else:
            base_priority = TicketPriority.LOW

        # Urgency signals
        urgency_signals = ['紧急', '急', '马上', '立即', 'urgent', 'asap', 'immediately', 'critical']
        if any(signal in text for signal in urgency_signals):
            base_priority = TicketPriority.URGENT

        # Celebrity/enterprise signals bump priority
        enterprise_signals = ['好莱坞', '明星', '经纪', 'sag', 'actor', 'agency', 'studio']
        if any(signal in text for signal in enterprise_signals) and category == TicketCategory.ENTERPRISE:
            base_priority = TicketPriority.HIGH

        return base_priority

    def _get_recommended_action(self, category: TicketCategory, priority: TicketPriority) -> str:
        """Get recommended action based on category and priority."""
        actions = {
            (TicketCategory.AUTH_ISSUE, TicketPriority.LOW): "send自助重置链接",
            (TicketCategory.AUTH_ISSUE, TicketPriority.MEDIUM): "审核后手动处理",
            (TicketCategory.AUTH_ISSUE, TicketPriority.HIGH): "立即处理",
            (TicketCategory.AUTH_ISSUE, TicketPriority.URGENT): "立即处理+确认用户",

            (TicketCategory.AUTHORIZATION, TicketPriority.LOW): "自动回复授权指南",
            (TicketCategory.AUTHORIZATION, TicketPriority.MEDIUM): "审核授权请求",
            (TicketCategory.AUTHORIZATION, TicketPriority.HIGH): "优先审核+联系权利人",
            (TicketCategory.AUTHORIZATION, TicketPriority.URGENT): "暂停相关授权+立即处理",

            (TicketCategory.DISPUTE, TicketPriority.LOW): "记录+定期审核",
            (TicketCategory.DISPUTE, TicketPriority.MEDIUM): "请求双方证据",
            (TicketCategory.DISPUTE, TicketPriority.HIGH): "法务介入评估",
            (TicketCategory.DISPUTE, TicketPriority.URGENT): "立即下架+法务+报警阈值",

            (TicketCategory.ENTERPRISE, TicketPriority.LOW): "自动回复企业介绍",
            (TicketCategory.ENTERPRISE, TicketPriority.MEDIUM): "安排商务对接",
            (TicketCategory.ENTERPRISE, TicketPriority.HIGH): "优先安排演示",
            (TicketCategory.ENTERPRISE, TicketPriority.URGENT): "立即安排高层通话",

            (TicketCategory.BUG_REPORT, TicketPriority.LOW): "记录到bug跟踪系统",
            (TicketCategory.BUG_REPORT, TicketPriority.MEDIUM): "复现并修复",
            (TicketCategory.BUG_REPORT, TicketPriority.HIGH): "紧急修复",
            (TicketCategory.BUG_REPORT, TicketPriority.URGENT): "hotfix立即发布",

            (TicketCategory.DATA_REQUEST, TicketPriority.LOW): "7天内处理",
            (TicketCategory.DATA_REQUEST, TicketPriority.MEDIUM): "5天内处理",
            (TicketCategory.DATA_REQUEST, TicketPriority.HIGH): "3天内处理",
            (TicketCategory.DATA_REQUEST, TicketPriority.URGENT): "24小时内处理",

            (TicketCategory.PAYMENT, TicketPriority.LOW): "核查后回复",
            (TicketCategory.PAYMENT, TicketPriority.MEDIUM): "财务审核",
            (TicketCategory.PAYMENT, TicketPriority.HIGH): "优先财务审核",
            (TicketCategory.PAYMENT, TicketPriority.URGENT): "立即核查+备用渠道",

            (TicketCategory.OTHER, TicketPriority.LOW): "自动分类失败-人工审核",
            (TicketCategory.OTHER, TicketPriority.MEDIUM): "人工审核",
            (TicketCategory.OTHER, TicketPriority.HIGH): "优先人工审核",
            (TicketCategory.OTHER, TicketPriority.URGENT): "立即人工处理",
        }

        return actions.get((category, priority), "人工审核")

    def create_ticket(self, subject: str, body: str, email: str = "",
                      source: str = "contact_form") -> Optional[int]:
        """Create an admin ticket from a message."""
        if not DB_AVAILABLE:
            logger.warning("Database not available, cannot create ticket")
            return None

        triage = self.triage_message(subject, body, email)

        conn, c, is_pg = get_db_conn()
        try:
            c.execute('''INSERT INTO tickets
                        (subject, body, email, source, category, priority, status, recommended_action, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)''',
                     (subject, body, email, source, triage['category'], triage['priority'],
                      'open', triage['action'], datetime.now().isoformat()))
            conn.commit()
            ticket_id = last_insert_id(conn, c, is_pg)
            logger.info(f"Created ticket {ticket_id}: {triage['category']} / {triage['priority']}")
            return ticket_id
        except Exception as e:
            logger.error(f"Failed to create ticket: {e}")
            return None
        finally:
            conn.close()

    def get_ticket_queue(self, status: str = "open",
                         priority: str = None,
                         category: str = None,
                         limit: int = 50) -> List[Dict]:
        """Get pending tickets for admin review."""
        if not DB_AVAILABLE:
            return []

        conn, c, is_pg = get_db_conn()
        query = "SELECT * FROM tickets WHERE status = %s"
        params = [status]

        if priority:
            query += " AND priority = %s"
            params.append(priority)
        if category:
            query += " AND category = %s"
            params.append(category)

        query += " ORDER BY CASE priority WHEN 'urgent' THEN 1 WHEN 'high' THEN 2 WHEN 'medium' THEN 3 ELSE 4 END, created_at DESC LIMIT %s"
        params.append(limit)

        c.execute(query, tuple(params))
        tickets = [dict(row) for row in c.fetchall()]
        conn.close()

        return tickets

    def get_system_health(self) -> Dict[str, Any]:
        """Get system health status."""
        health = {
            "timestamp": datetime.now().isoformat(),
            "status": "healthy",
            "checks": {}
        }

        if not DB_AVAILABLE:
            health["checks"]["database"] = {"status": "unknown", "message": "DB not available"}
            health["status"] = "degraded"
            return health

        try:
            conn, c, is_pg = get_db_conn()

            # Check database connectivity
            c.execute("SELECT 1")
            health["checks"]["database"] = {"status": "ok", "message": "Connected"}

            # Check user count
            c.execute("SELECT COUNT(*) as count FROM users")
            users_count = dict(c.fetchone())['count']
            health["checks"]["users"] = {"status": "ok", "count": users_count}

            # Check faces count
            c.execute("SELECT COUNT(*) as count FROM faces WHERE status='active'")
            faces_count = dict(c.fetchone())['count']
            health["checks"]["faces"] = {"status": "ok", "count": faces_count}

            # Check fingerprints count
            try:
                c.execute("SELECT COUNT(*) as count FROM portrait_fingerprints")
                fp_count = dict(c.fetchone())['count']
                health["checks"]["fingerprints"] = {"status": "ok", "count": fp_count}
            except Exception:
                health["checks"]["fingerprints"] = {"status": "not_initialized", "count": 0}

            # Check open tickets
            try:
                c.execute("SELECT COUNT(*) as count FROM tickets WHERE status='open'")
                open_tickets = dict(c.fetchone())['count']
                health["checks"]["tickets"] = {"status": "ok" if open_tickets < 10 else "attention", "open_count": open_tickets}
            except Exception:
                health["checks"]["tickets"] = {"status": "not_initialized", "open_count": 0}

            conn.close()

            # Overall status
            critical_failures = [k for k, v in health["checks"].items()
                               if v.get("status") not in ["ok", "not_initialized", "unknown"]]
            if critical_failures:
                health["status"] = "unhealthy"

        except Exception as e:
            health["status"] = "unhealthy"
            health["checks"]["database"] = {"status": "error", "message": str(e)}

        return health

    def generate_daily_report(self) -> Dict[str, Any]:
        """Generate daily operations report."""
        report = {
            "date": datetime.now().date().isoformat(),
            "report_type": "daily",
            "generated_at": datetime.now().isoformat()
        }

        if not DB_AVAILABLE:
            return self._generate_mock_report(report)

        conn, c, is_pg = get_db_conn()

        try:
            today = datetime.now().date().isoformat()

            # New registrations today
            c.execute("SELECT COUNT(*) as count FROM users WHERE DATE(created_at) = %s", (today,))
            report["new_registrations"] = dict(c.fetchone())['count']

            # New faces registered today
            c.execute("SELECT COUNT(*) as count FROM faces WHERE DATE(created_at) = %s", (today,))
            report["new_faces"] = dict(c.fetchone())['count']

            # Search queries today
            try:
                c.execute("SELECT COUNT(*) as count FROM search_queries WHERE DATE(created_at) = %s", (today,))
                report["search_queries"] = dict(c.fetchone())['count']
            except Exception:
                report["search_queries"] = 0

            # Revenue today (sum of completed transactions)
            c.execute("""SELECT COALESCE(SUM(amount), 0) as total
                        FROM transactions
                        WHERE DATE(created_at) = %s AND status = 'completed'""", (today,))
            report["revenue_today"] = dict(c.fetchone())['total']

            # Open tickets
            c.execute("SELECT COUNT(*) as count FROM tickets WHERE status = 'open'")
            report["open_tickets"] = dict(c.fetchone())['count']

            # Urgent tickets
            c.execute("SELECT COUNT(*) as count FROM tickets WHERE status = 'open' AND priority = 'urgent'")
            report["urgent_tickets"] = dict(c.fetchone())['count']

            conn.close()

            # Add action items
            report["action_items"] = []
            if report.get("urgent_tickets", 0) > 0:
                report["action_items"].append(f"⚠️ {report['urgent_tickets']} 个紧急工单需要处理")
            if report.get("open_tickets", 0) > 5:
                report["action_items"].append(f"📋 {report['open_tickets']} 个开放工单待处理")
            if report.get("new_registrations", 0) == 0:
                report["action_items"].append("📢 今日无新注册，可能需要推广")

            report["status"] = "generated"

        except Exception as e:
            logger.error(f"Failed to generate daily report: {e}")
            conn.close()
            return self._generate_mock_report(report)

        return report

    def _generate_mock_report(self, report: Dict) -> Dict:
        """Generate a mock report when DB is not available."""
        report["new_registrations"] = 0
        report["new_faces"] = 0
        report["search_queries"] = 0
        report["revenue_today"] = 0
        report["open_tickets"] = 0
        report["urgent_tickets"] = 0
        report["action_items"] = ["系统正在初始化中"]
        report["status"] = "mock_data"
        return report

    def generate_weekly_report(self) -> Dict[str, Any]:
        """Generate weekly aggregated report."""
        report = {
            "date_range": {
                "start": (datetime.now() - timedelta(days=7)).date().isoformat(),
                "end": datetime.now().date().isoformat()
            },
            "report_type": "weekly",
            "generated_at": datetime.now().isoformat()
        }

        if not DB_AVAILABLE:
            return {**report, "status": "mock_data", "new_registrations": 0, "new_faces": 0}

        conn, c, is_pg = get_db_conn()

        try:
            start_date = (datetime.now() - timedelta(days=7)).date().isoformat()
            end_date = datetime.now().date().isoformat()

            # Weekly new registrations
            c.execute("""SELECT COUNT(*) as count FROM users
                        WHERE DATE(created_at) BETWEEN %s AND %s""", (start_date, end_date))
            report["new_registrations"] = dict(c.fetchone())['count']

            # Weekly new faces
            c.execute("""SELECT COUNT(*) as count FROM faces
                        WHERE DATE(created_at) BETWEEN %s AND %s""", (start_date, end_date))
            report["new_faces"] = dict(c.fetchone())['count']

            # Weekly search queries
            try:
                c.execute("""SELECT COUNT(*) as count FROM search_queries
                            WHERE DATE(created_at) BETWEEN %s AND %s""", (start_date, end_date))
                report["search_queries"] = dict(c.fetchone())['count']
            except Exception:
                report["search_queries"] = 0

            # Weekly revenue
            c.execute("""SELECT COALESCE(SUM(amount), 0) as total
                        FROM transactions
                        WHERE DATE(created_at) BETWEEN %s AND %s AND status = 'completed'""",
                     (start_date, end_date))
            report["revenue_week"] = dict(c.fetchone())['total']

            # Top categories this week
            try:
                c.execute("""SELECT category, COUNT(*) as count FROM tickets
                            WHERE DATE(created_at) BETWEEN %s AND %s
                            GROUP BY category ORDER BY count DESC LIMIT 5""",
                          (start_date, end_date))
                report["top_ticket_categories"] = [dict(row) for row in c.fetchall()]
            except Exception:
                report["top_ticket_categories"] = []

            conn.close()
            report["status"] = "generated"

        except Exception as e:
            logger.error(f"Failed to generate weekly report: {e}")
            conn.close()
            report["status"] = "error"
            report["error"] = str(e)

        return report

    def generate_monthly_report(self) -> Dict[str, Any]:
        """Generate monthly aggregated report."""
        report = {
            "date_range": {
                "start": (datetime.now().replace(day=1)).date().isoformat(),
                "end": datetime.now().date().isoformat()
            },
            "report_type": "monthly",
            "generated_at": datetime.now().isoformat()
        }

        if not DB_AVAILABLE:
            return {**report, "status": "mock_data"}

        conn, c, is_pg = get_db_conn()

        try:
            start_date = (datetime.now().replace(day=1)).date().isoformat()
            end_date = datetime.now().date().isoformat()

            # Monthly totals
            c.execute("""SELECT COUNT(*) as count FROM users
                        WHERE DATE(created_at) BETWEEN %s AND %s""", (start_date, end_date))
            report["new_registrations"] = dict(c.fetchone())['count']

            c.execute("""SELECT COUNT(*) as count FROM faces
                        WHERE DATE(created_at) BETWEEN %s AND %s""", (start_date, end_date))
            report["new_faces"] = dict(c.fetchone())['count']

            # Total users and faces
            c.execute("SELECT COUNT(*) as count FROM users")
            report["total_users"] = dict(c.fetchone())['count']

            c.execute("SELECT COUNT(*) as count FROM faces WHERE status='active'")
            report["total_active_faces"] = dict(c.fetchone())['count']

            # Monthly revenue
            c.execute("""SELECT COALESCE(SUM(amount), 0) as total
                        FROM transactions
                        WHERE DATE(created_at) BETWEEN %s AND %s AND status = 'completed'""",
                     (start_date, end_date))
            report["revenue_month"] = dict(c.fetchone())['total']

            # Platform fee collected
            try:
                c.execute("""SELECT COALESCE(SUM(platform_fee), 0) as total
                            FROM revenues
                            WHERE DATE(created_at) BETWEEN %s AND %s""",
                         (start_date, end_date))
                report["platform_fees"] = dict(c.fetchone())['total']
            except Exception:
                report["platform_fees"] = 0

            conn.close()
            report["status"] = "generated"

        except Exception as e:
            logger.error(f"Failed to generate monthly report: {e}")
            conn.close()
            report["status"] = "error"

        return report


# Singleton instance
admin_automation = AdminAutomation()
