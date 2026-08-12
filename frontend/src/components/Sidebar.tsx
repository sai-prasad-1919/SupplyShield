import React from 'react';
import { Link, useLocation } from '@tanstack/react-router';
import { TruckIcon, FactoryIcon, ActivityIcon, InfoIcon, SettingsIcon, ShieldCheckIcon, BrainIcon, ClipboardIcon } from './Icons';

export const Sidebar = () => {
  const location = useLocation();

  const links = [
    { to: '/dashboard', label: 'Shipments', icon: TruckIcon },
    { to: '/suppliers', label: 'Suppliers', icon: FactoryIcon },
    { to: '/demand', label: 'Demand', icon: ActivityIcon },
    { to: '/guidance', label: 'Guidance', icon: InfoIcon },
    { to: '/federated', label: 'FL Model', icon: BrainIcon },
    { to: '/feedback', label: 'Feedback', icon: ClipboardIcon },
  ];

  return (
    <aside className="w-64 bg-[var(--sidebar)] border-r border-border h-screen sticky top-0 hidden md:flex flex-col">
      <div className="p-6 flex items-center space-x-3 text-white">
        <ShieldCheckIcon className="w-6 h-6 text-primary" />
        <span className="font-bold text-lg tracking-tight">SupplyShield</span>
      </div>
      
      <nav className="flex-1 px-4 space-y-1">
        {links.map((link) => {
          const isActive = location.pathname.startsWith(link.to);
          return (
            <Link
              key={link.to}
              to={link.to}
              className={`flex items-center space-x-3 px-3 py-2.5 rounded-md transition-colors ${
                isActive 
                  ? 'bg-[var(--sidebar-accent)] text-white font-medium' 
                  : 'text-slate-400 hover:text-white hover:bg-[var(--sidebar-accent)]'
              }`}
            >
              <link.icon className="w-5 h-5" />
              <span>{link.label}</span>
            </Link>
          );
        })}
      </nav>
      
      <div className="p-4 border-t border-border">
        <Link
          to="/dashboard"
          className="flex items-center space-x-3 px-3 py-2.5 rounded-md transition-colors text-slate-400 hover:text-white hover:bg-[var(--sidebar-accent)]"
        >
          <SettingsIcon className="w-5 h-5" />
          <span>Settings</span>
        </Link>
      </div>
    </aside>
  );
};
