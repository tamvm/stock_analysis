#!/usr/bin/env python3

"""
Database Migration Runner
Runs SQL migrations in order and tracks applied migrations.

Usage:
    python run_migrations.py              # Run all pending migrations
    python run_migrations.py --status     # Show migration status
    python run_migrations.py --help       # Show help
"""

import sqlite3
import argparse
import sys
from pathlib import Path
from datetime import datetime

class MigrationRunner:
    def __init__(self, db_path="db/investment_data.db", migrations_path="migrations"):
        self.db_path = db_path
        self.migrations_path = Path(migrations_path)
        
    def setup_migrations_table(self):
        """Create schema_migrations table if it doesn't exist"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS schema_migrations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                migration_name TEXT NOT NULL UNIQUE,
                applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
        
    def get_applied_migrations(self):
        """Get list of already applied migrations"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute('SELECT migration_name FROM schema_migrations ORDER BY migration_name')
        applied = [row[0] for row in cursor.fetchall()]
        conn.close()
        return applied
        
    def get_migration_files(self):
        """Get list of all migration files"""
        if not self.migrations_path.exists():
            print(f"❌ Migrations directory not found: {self.migrations_path}")
            return []
            
        migration_files = sorted(self.migrations_path.glob("*.sql"))
        return migration_files
        
    def apply_migration(self, migration_file):
        """Apply a single migration file"""
        migration_name = migration_file.name
        
        print(f"📝 Applying migration: {migration_name}")
        
        try:
            # Read migration SQL
            with open(migration_file, 'r') as f:
                sql = f.read()
            
            # Execute migration
            conn = sqlite3.connect(self.db_path)
            conn.executescript(sql)
            
            # Record migration as applied
            conn.execute(
                'INSERT INTO schema_migrations (migration_name) VALUES (?)',
                (migration_name,)
            )
            
            conn.commit()
            conn.close()
            
            print(f"✅ Successfully applied: {migration_name}")
            return True
            
        except Exception as e:
            print(f"❌ Error applying {migration_name}: {e}")
            return False
            
    def run_migrations(self):
        """Run all pending migrations"""
        self.setup_migrations_table()
        
        applied = self.get_applied_migrations()
        migration_files = self.get_migration_files()
        
        if not migration_files:
            print("⚠️  No migration files found")
            return
            
        pending = [f for f in migration_files if f.name not in applied]
        
        if not pending:
            print("✅ All migrations are up to date")
            return
            
        print(f"\n🚀 Found {len(pending)} pending migration(s)\n")
        
        success_count = 0
        for migration_file in pending:
            if self.apply_migration(migration_file):
                success_count += 1
            else:
                print(f"\n❌ Migration failed. Stopping here.")
                break
                
        print(f"\n✅ Applied {success_count} migration(s) successfully")
        
    def show_status(self):
        """Show migration status"""
        self.setup_migrations_table()
        
        applied = self.get_applied_migrations()
        migration_files = self.get_migration_files()
        
        print("\n📊 Migration Status\n")
        print("-" * 60)
        
        if not migration_files:
            print("⚠️  No migration files found")
            return
            
        for migration_file in migration_files:
            status = "✅ Applied" if migration_file.name in applied else "⏳ Pending"
            print(f"{status:12} {migration_file.name}")
            
        print("-" * 60)
        print(f"Total: {len(migration_files)} migrations, {len(applied)} applied, {len(migration_files) - len(applied)} pending")
        print()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Database migration runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python run_migrations.py              # Run all pending migrations
  python run_migrations.py --status     # Show migration status
        '''
    )
    
    parser.add_argument('--status', action='store_true', help='Show migration status')
    parser.add_argument('--db', default='db/investment_data.db', help='Database file path')
    parser.add_argument('--migrations', default='migrations', help='Migrations directory path')
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    
    runner = MigrationRunner(db_path=args.db, migrations_path=args.migrations)
    
    if args.status:
        runner.show_status()
    else:
        runner.run_migrations()

if __name__ == "__main__":
    main()
