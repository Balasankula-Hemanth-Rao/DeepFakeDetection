import { useKeyboardShortcuts } from '@/hooks/useKeyboardShortcuts';

export const KeyboardShortcutsProvider = ({ children }: { children: React.ReactNode }) => {
  useKeyboardShortcuts();
  return <>{children}</>;
};
