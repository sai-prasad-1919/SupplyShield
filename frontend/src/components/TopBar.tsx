import React from 'react';
import { useNavigate } from '@tanstack/react-router';
import { SearchIcon, BellIcon, LogOutIcon } from './Icons';
import { clearSession } from '../lib/api';

export const TopBar = () => {
  const navigate = useNavigate();
  const orgName = localStorage.getItem('supplyshield_org_name') || 'Organization';
  const ownerName = localStorage.getItem('supplyshield_owner_name') || '';
  const orgId = localStorage.getItem('supplyshield_org_id') || '';

  const handleLogout = () => {
    clearSession();
    navigate({ to: '/login' });
  };

  // Initials from owner name
  const initials = ownerName
    .split(' ')
    .map(w => w[0])
    .slice(0, 2)
    .join('')
    .toUpperCase() || 'U';

  return (
    <header className="h-16 border-b border-border bg-background flex items-center justify-between px-6 sticky top-0 z-10">
      <div className="flex items-center space-x-3">
        <div>
          <h2 className="font-semibold text-white text-sm hidden sm:block">{orgName}</h2>
          {orgId && (
            <span className="text-xs text-slate-500 font-mono hidden sm:block">{orgId}</span>
          )}
        </div>
      </div>

      <div className="flex items-center space-x-4">
        <div className="relative hidden sm:block">
          <SearchIcon className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
          <input
            type="text"
            placeholder="Search shipments..."
            className="bg-card border border-border rounded-full pl-9 pr-4 py-1.5 text-sm focus:outline-none focus:border-primary text-white placeholder-slate-500 w-64"
          />
        </div>

        <button className="relative p-2 text-slate-400 hover:text-white transition-colors">
          <BellIcon className="w-5 h-5" />
          <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-risk-high rounded-full border border-background"></span>
        </button>

        <div
          className="h-8 w-8 rounded-full bg-primary/20 flex items-center justify-center text-primary font-semibold text-xs border border-primary/30"
          title={ownerName}
        >
          {initials}
        </div>

        <button
          onClick={handleLogout}
          className="p-2 text-slate-400 hover:text-white transition-colors"
          title="Log out"
          aria-label="Log out"
        >
          <LogOutIcon className="w-5 h-5" />
        </button>
      </div>
    </header>
  );
};
