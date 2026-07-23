import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from '@tanstack/react-router';
import { SearchIcon, BellIcon, LogOutIcon } from './Icons';
import { clearSession } from '../lib/api';

export const TopBar = () => {
  const navigate = useNavigate();
  const [profileOpen, setProfileOpen] = useState(false);
  const profileRef = useRef<HTMLDivElement>(null);

  const orgName  = localStorage.getItem('supplyshield_org_name')  || 'Organization';
  const ownerName = localStorage.getItem('supplyshield_owner_name') || '';
  const orgId    = localStorage.getItem('supplyshield_org_id')    || '';

  // Derive org type from the org key stored in token (Retail / Manufacturing / Logistics)
  const orgTypeMap: Record<string, string> = {
    'NVM': 'Retail',
    'TIT': 'Manufacturing',
    'SWL': 'Logistics',
  };
  const prefix  = orgId.split('-')[0] || '';
  const orgType = orgTypeMap[prefix] || 'Organisation';

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

  // Close dropdown on outside click
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (profileRef.current && !profileRef.current.contains(e.target as Node)) {
        setProfileOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    // Solid background using exact CSS variable value — avoids Tailwind v4 bg-background resolution issues
    <header
      className="h-16 border-b border-slate-700/60 flex items-center justify-between px-6 sticky top-0 z-30"
      style={{ backgroundColor: 'oklch(0.17 0.015 260)' }}
    >
      {/* Left — org name + ID */}
      <div className="flex items-center space-x-3">
        <div>
          <h2 className="font-semibold text-white text-sm hidden sm:block">{orgName}</h2>
          {orgId && (
            <span className="text-xs text-slate-500 font-mono hidden sm:block">{orgId}</span>
          )}
        </div>
      </div>

      {/* Right — search, bell, avatar, logout */}
      <div className="flex items-center space-x-4">

        {/* Search — placeholder for future shipment filtering from header */}
        <div className="relative hidden sm:block">
          <SearchIcon className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
          <input
            id="topbar-search"
            type="text"
            placeholder="Search shipments..."
            className="border border-slate-700 rounded-full pl-9 pr-4 py-1.5 text-sm focus:outline-none focus:border-blue-500 text-white placeholder-slate-500 w-64"
            style={{ backgroundColor: 'oklch(0.22 0.018 260)' }}
          />
        </div>

        {/* Bell — placeholder for future real-time risk alerts feed */}
        <button
          id="topbar-notifications"
          className="relative p-2 text-slate-400 hover:text-white transition-colors"
          title="Notifications (coming soon)"
        >
          <BellIcon className="w-5 h-5" />
          <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-red-500 rounded-full border border-slate-900" />
        </button>

        {/* Avatar + Profile Dropdown */}
        <div ref={profileRef} className="relative">
          <button
            id="topbar-profile"
            onClick={() => setProfileOpen(prev => !prev)}
            className="h-8 w-8 rounded-full flex items-center justify-center font-semibold text-xs border transition-all cursor-pointer hover:ring-2 hover:ring-blue-500/50"
            style={{
              backgroundColor: 'oklch(0.65 0.19 255 / 0.2)',
              borderColor: 'oklch(0.65 0.19 255 / 0.3)',
              color: 'oklch(0.65 0.19 255)',
            }}
            title={`${ownerName} — click to view profile`}
            aria-label="View organisation profile"
            aria-expanded={profileOpen}
          >
            {initials}
          </button>

          {/* Profile Dropdown Panel */}
          {profileOpen && (
            <div
              className="absolute right-0 mt-2 w-72 rounded-xl border border-slate-700/80 shadow-2xl overflow-hidden z-50"
              style={{ backgroundColor: 'oklch(0.22 0.018 260)' }}
            >
              {/* Header strip */}
              <div
                className="px-5 py-4 border-b border-slate-700/60 flex items-center space-x-3"
                style={{ backgroundColor: 'oklch(0.19 0.016 260)' }}
              >
                <div
                  className="h-10 w-10 rounded-full flex items-center justify-center font-bold text-sm shrink-0"
                  style={{
                    backgroundColor: 'oklch(0.65 0.19 255 / 0.2)',
                    color: 'oklch(0.65 0.19 255)',
                    border: '1px solid oklch(0.65 0.19 255 / 0.3)',
                  }}
                >
                  {initials}
                </div>
                <div className="overflow-hidden">
                  <p className="text-white font-semibold text-sm truncate">{ownerName}</p>
                  <p className="text-slate-400 text-xs truncate">{orgType} · Owner</p>
                </div>
              </div>

              {/* Details */}
              <div className="px-5 py-4 space-y-3">
                <ProfileRow label="Organisation" value={orgName} />
                <ProfileRow label="Org ID" value={orgId} mono />
                <ProfileRow label="Type" value={orgType} />
                <ProfileRow label="Role" value="Owner" />
              </div>

              {/* Footer note */}
              <div
                className="px-5 py-3 border-t border-slate-700/60 text-xs text-slate-500 italic"
                style={{ backgroundColor: 'oklch(0.19 0.016 260)' }}
              >
                Password change available after email authentication (coming soon)
              </div>
            </div>
          )}
        </div>

        {/* Logout */}
        <button
          id="topbar-logout"
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

// Small helper row component
const ProfileRow = ({
  label,
  value,
  mono = false,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) => (
  <div className="flex justify-between items-center gap-4">
    <span className="text-slate-500 text-xs shrink-0">{label}</span>
    <span
      className={`text-white text-xs text-right truncate ${mono ? 'font-mono' : 'font-medium'}`}
    >
      {value || '—'}
    </span>
  </div>
);
